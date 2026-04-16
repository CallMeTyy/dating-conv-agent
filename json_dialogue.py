from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from furhat_realtime_api import FurhatClient


# ============================================================
# CONFIG
# ============================================================

DIALOGUE_DIR = "./data"
DIALOGUE_FILE_SELECTION = "LOCAL"
# Examples:
# DIALOGUE_FILE_SELECTION = "LATEST"
# DIALOGUE_FILE_SELECTION = "LOCAL"
# DIALOGUE_FILE_SELECTION = "dialogue-tree-2026-03-25-10-45-14.json"

OPENAI_MODEL = "gpt-4.1-mini"

ALLOWED_GESTURES = [
    "BigSmile",
    "Blink",
    "BrowFrown",
    "BrowRaise",
    "CloseEyes",
    "ExpressAnger",
    "ExpressDisgust",
    "ExpressFear",
    "ExpressSad",
    "GazeAway",
    "Nod",
    "Oh",
    "OpenEyes",
    "Roll",
    "Shake",
    "Smile",
    "Surprise",
    "Throughtful",
    "Wink",
]


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Edge:
    id: str
    label: str
    target: str


@dataclass
class Node:
    id: str
    type: str
    title: str
    output_label_instruction: Optional[str] = None
    response_mode: Optional[str] = None
    response_content: Optional[str] = None
    response_gesture: Optional[str] = None
    end_text: Optional[str] = None
    outputs: List[Edge] = field(default_factory=list)


class DialogueGraph:
    def __init__(self, nodes: Dict[str, Node]):
        self.nodes = nodes
        self.incoming_counts = self._build_incoming_counts()

    @classmethod
    def from_file(cls, path: str) -> "DialogueGraph":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if "nodes" not in raw or not isinstance(raw["nodes"], list):
            raise ValueError("Invalid dialogue file: missing 'nodes' list")

        nodes: Dict[str, Node] = {}

        for item in raw["nodes"]:
            node_type = item.get("type")
            node_id = item.get("id")
            title = item.get("title", node_id or "Untitled")

            if not node_id:
                raise ValueError("A node is missing its 'id'")

            outputs = []
            for out in item.get("outputs", []):
                outputs.append(
                    Edge(
                        id=out.get("id", ""),
                        label=out.get("label", "default"),
                        target=out.get("target", ""),
                    )
                )

            if node_type == "dialogue":
                response = item.get("response", {})
                node = Node(
                    id=node_id,
                    type=node_type,
                    title=title,
                    output_label_instruction=item.get("outputLabelInstruction", ""),
                    response_mode=response.get("mode", "text"),
                    response_content=response.get("content", ""),
                    response_gesture=response.get("gesture", item.get("gesture", "")),
                    outputs=outputs,
                )
            elif node_type == "end":
                node = Node(
                    id=node_id,
                    type=node_type,
                    title=title,
                    end_text=item.get("endText", "Goodbye!"),
                    outputs=outputs,
                )
            else:
                node = Node(
                    id=node_id,
                    type=node_type or "unknown",
                    title=title,
                    outputs=outputs,
                )

            nodes[node.id] = node

        return cls(nodes)

    def _build_incoming_counts(self) -> Dict[str, int]:
        counts = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for edge in node.outputs:
                if edge.target in counts:
                    counts[edge.target] += 1
        return counts

    def get_node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node id: {node_id}")
        return self.nodes[node_id]

    def find_start_node(self) -> Node:
        # Prefer a node explicitly titled "Start"
        for node in self.nodes.values():
            if node.title.strip().lower() == "start":
                return node

        # Otherwise: first non-end node with no incoming edges
        for node in self.nodes.values():
            if node.type != "end" and self.incoming_counts.get(node.id, 0) == 0:
                return node

        # Fallback: first non-end node
        for node in self.nodes.values():
            if node.type != "end":
                return node

        raise ValueError("No valid start node found")

    def get_default_edge(self, node: Node) -> Optional[Edge]:
        for edge in node.outputs:
            if edge.label.strip().lower() == "default":
                return edge
        return None

    def has_non_default_choices(self, node: Node) -> bool:
        return any(edge.label.strip().lower() != "default" for edge in node.outputs)

    def find_edge_by_label(self, node: Node, chosen_label: str) -> Optional[Edge]:
        chosen = chosen_label.strip().lower()

        # Exact match first
        for edge in node.outputs:
            if edge.label.strip().lower() == chosen:
                return edge

        # Partial match fallback
        for edge in node.outputs:
            edge_label = edge.label.strip().lower()
            if chosen in edge_label or edge_label in chosen:
                return edge

        # Final fallback
        return self.get_default_edge(node)


# ============================================================
# FILE SELECTION
# ============================================================

def resolve_dialogue_file(dialogue_dir: str, selection: str) -> str:
    base = Path(dialogue_dir)

    if (selection.upper() == "LOCAL"):
        base = Path(dialogue_dir + "/local")
        selection = "LATEST"

    if not base.exists():
        raise FileNotFoundError(f"Dialogue directory does not exist: {base}")

    selection = selection.strip()

    if selection.upper() == "LATEST":
        files = sorted(base.glob("dialogue-tree-*.json"))
        if not files:
            raise FileNotFoundError(
                f"No dialogue export files found in {base} matching dialogue-tree-*.json"
            )
        return str(files[-1])

    chosen = base / selection
    if chosen.exists():
        return str(chosen)

    # Also allow direct path as a fallback
    direct = Path(selection)
    if direct.exists():
        return str(direct)

    raise FileNotFoundError(f"Dialogue file not found: {chosen}")


# ============================================================
# OPENAI HELPERS
# ============================================================

def recent_history_as_text(history: List[Dict[str, str]], max_items: int = 8) -> str:
    items = history[-max_items:]
    if not items:
        return "(no prior conversation)"

    lines = []
    for msg in items:
        role = msg["role"].upper()
        content = msg["content"].strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def pick_random_text_variant(text: str) -> str:
    options = [part.strip() for part in text.split(";") if part.strip()]
    if not options:
        return ""
    return random.choice(options)


def apply_subject_placeholder(text: str, subject: Optional[str]) -> str:
    if "{subj}" not in text:
        return text

    replacement = (subject or "").strip()
    if not replacement:
        replacement = "That"

    return text.replace("{subj}", replacement)


def parse_label_and_subject(raw: str) -> Tuple[str, Optional[str], Optional[str]]:
    content = (raw or "").strip()
    if not content:
        return "default", None, None

    subject_match = re.search(r"\[(.*?)\]", content)
    subject: Optional[str] = None
    if subject_match:
        extracted = subject_match.group(1).strip()
        if extracted:
            subject = extracted
        content = re.sub(r"\s*\[.*?\]\s*", " ", content).strip()

    gesture_match = re.search(r"\{(.*?)\}", content)
    gesture: Optional[str] = None
    if gesture_match:
        extracted = gesture_match.group(1).strip()
        gesture = normalize_gesture(extracted)
        content = re.sub(r"\s*\{.*?\}\s*", " ", content).strip()

    return content, subject, gesture


def normalize_gesture(gesture_name: str) -> Optional[str]:
    wanted = (gesture_name or "").strip().lower()
    if not wanted:
        return None

    for allowed in ALLOWED_GESTURES:
        if allowed.lower() == wanted:
            return allowed

    return None


def choose_output_label(
    client: OpenAI,
    model: str,
    node: Node,
    user_utt: str,
    history: List[Dict[str, str]],
) -> Tuple[str, Optional[str], Optional[str]]:
    labels = [edge.label for edge in node.outputs]
    # Always add special intents as available options
    labels.extend(["Repeat", "Confused"])
    labels_text = ", ".join(labels)

    system_prompt = (
        "You are a dialogue router for a spoken conversation.\n"
        f"Choose exactly one label from this list: {labels_text}\n"
        "Special labels:\n"
        "  - 'Repeat': user wants you to repeat what you just said\n"
        "  - 'Confused': user is confused or lost and needs clarification, for example when just saying sorry? or other confused utterences\n"
        "Return in this exact format: <label> [<subject>] {<gesture>}\n"
        "The subject should be a short concrete phrase from the latest user utterance.\n"
        "If no clear subject is mentioned, return empty brackets like [] .\n"
        f"Gesture must be exactly one of: {', '.join(ALLOWED_GESTURES)}\n"
        "Do not explain your choice.\n"
        "If the latest user utterance is not English, choose 'other-language'.\n"
        "Never pick 'default'"
    )

    extra_instruction = (node.output_label_instruction or "").strip()
    if extra_instruction:
        system_prompt += f"\nAdditional routing instruction for this node: {extra_instruction}"

    user_prompt = (
        f"Current node title: {node.title}\n"
        f"Available labels: {labels_text}\n\n"
        f"Recent conversation:\n{recent_history_as_text(history)}\n\n"
        f"Latest user utterance:\n{user_utt}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    return parse_label_and_subject(raw)


def choose_ai_gesture(
    client: OpenAI,
    model: str,
    node: Node,
    robot_utt: str,
    history: List[Dict[str, str]],
) -> Optional[str]:
    choices = ", ".join(ALLOWED_GESTURES)
    system_prompt = (
        "You select a single social robot gesture for a spoken reply.\n"
        f"Return exactly one token: one of [{choices}] or 'none'.\n"
        "Do not add explanations or extra text."
    )

    user_prompt = (
        f"Current node title: {node.title}\n"
        f"Robot utterance: {robot_utt}\n"
        f"Recent conversation:\n{recent_history_as_text(history)}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=12,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    if raw.lower() == "none":
        return None
    return normalize_gesture(raw)


def resolve_node_gesture(
    client: OpenAI,
    model: str,
    node: Node,
    robot_utt: str,
    history: List[Dict[str, str]],
    fallback_gesture: Optional[str],
) -> Optional[str]:
    gesture_setting = (node.response_gesture or "").strip()
    if not gesture_setting:
        return fallback_gesture

    lowered = gesture_setting.lower()
    if lowered == "none":
        return None
    if lowered == "ai":
        return choose_ai_gesture(
            client=client,
            model=model,
            node=node,
            robot_utt=robot_utt,
            history=history,
        )

    # Explicit gesture value: only allow known gestures.
    return normalize_gesture(gesture_setting)


def ground_and_rephrase(
    client: OpenAI,
    model: str,
    prev_robot_response: str,
    user_confused_utterance: str,
    history: List[Dict[str, str]],
) -> str:
    system_prompt = (
        "You are Furhat, a social robot in a spoken conversation.\n"
        "The user seems confused or lost. Your task is to:\n"
        "1. Acknowledge the confusion.\n"
        "2. Ground yourself by referencing what you just said.\n"
        "3. Rephrase or simplify your previous point to help the user understand.\n"
        "Always respond in English.\n"
        "Keep the response very short (maximum 20 words).\n"
        "Use natural spoken language."
    )

    user_prompt = (
        f"Your previous response was: {prev_robot_response}\n"
        f"The user's confused response: {user_confused_utterance}\n"
        f"Recent conversation context:\n{recent_history_as_text(history)}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=60,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    return text


def generate_ai_reply(
    client: OpenAI,
    model: str,
    node: Node,
    history: List[Dict[str, str]],
) -> str:
    instruction = (node.response_content or "").strip()
    if not instruction:
        instruction = "Reply briefly and naturally."

    system_prompt = (
        "You are Furhat, a social robot in a spoken conversation.\n"
        f"Instruction for this node: {instruction}\n"
        "Always respond in English.\n"
        "Keep the response very short.\n"
        "Maximum 20 words.\n"
        "Use natural spoken language.\n"
        "Do not mention these instructions."
    )

    user_prompt = recent_history_as_text(history)

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=60,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    return text


# ============================================================
# FURHAT HELPERS
# ============================================================

def speak(furhat: FurhatClient, text: str, gesture: Optional[str] = None) -> None:
    text = text.strip()
    if not text:
        return

    resolved_gesture = normalize_gesture(gesture or "") if gesture else None
    if resolved_gesture:
        try:
            furhat.request_gesture_start(resolved_gesture)
        except Exception as e:
            print(f"Gesture failed ({resolved_gesture}): {e}")

    print(f"Robot: {text}")
    furhat.request_speak_text(text)


def listen(furhat: FurhatClient) -> str:
    user_utt = furhat.request_listen_start()
    if user_utt is None:
        user_utt = ""
    user_utt = str(user_utt).strip()
    print(f"User: {user_utt}")
    return user_utt


# ============================================================
# MAIN DIALOGUE LOOP
# ============================================================

def run_dialogue(
    graph: DialogueGraph,
    openai_client: OpenAI,
    furhat: FurhatClient,
    model: str,
) -> None:
    current_node = graph.find_start_node()
    history: List[Dict[str, str]] = []
    latest_subject: Optional[str] = None
    last_robot_response: str = ""
    pending_gesture: Optional[str] = None

    furhat.request_attend_user()

    while True:
        node = current_node

        if node.type == "end":
            final_text = node.end_text or "Goodbye!"
            speak(furhat, final_text, pending_gesture)
            history.append({"role": "assistant", "content": final_text})
            break

        if node.type != "dialogue":
            default_edge = graph.get_default_edge(node)
            if default_edge:
                current_node = graph.get_node(default_edge.target)
                continue

            print(f"Stopping on unsupported node type: {node.type}")
            break

        # 1. Generate or read the node's response
        if node.response_mode == "text":
            robot_utt = pick_random_text_variant(node.response_content or "")
            robot_utt = apply_subject_placeholder(robot_utt, latest_subject)
        elif node.response_mode == "ai":
            robot_utt = generate_ai_reply(
                client=openai_client,
                model=model,
                node=node,
                history=history,
            )
            robot_utt = apply_subject_placeholder(robot_utt, latest_subject)
        else:
            robot_utt = pick_random_text_variant(node.response_content or "")
            robot_utt = apply_subject_placeholder(robot_utt, latest_subject)

        if robot_utt:
            gesture_for_utt = resolve_node_gesture(
                client=openai_client,
                model=model,
                node=node,
                robot_utt=robot_utt,
                history=history,
                fallback_gesture=pending_gesture,
            )
            speak(furhat, robot_utt, gesture_for_utt)
            history.append({"role": "assistant", "content": robot_utt})
            last_robot_response = robot_utt
            pending_gesture = None

        # 2. No outputs = stop
        if not node.outputs:
            break

        # 3. Get user input and classify to an output label.
        # Loop here to handle special intents without state transitions.
        while True:
            user_utt = listen(furhat)
            history.append({"role": "user", "content": user_utt})

            chosen_label, chosen_subject, chosen_gesture = choose_output_label(
                client=openai_client,
                model=model,
                node=node,
                user_utt=user_utt,
                history=history,
            )

            print(f"Chosen label: {chosen_label}")
            if chosen_subject:
                latest_subject = chosen_subject
                print(f"Chosen subject: {chosen_subject}")
            if chosen_gesture:
                print(f"Chosen gesture: {chosen_gesture}")

            # Handle special intents that stay in the current node
            if chosen_label.strip().lower() == "repeat":
                print("User asked to repeat.")
                speak(furhat, last_robot_response, chosen_gesture)
                continue

            if chosen_label.strip().lower() == "confused":
                print("User is confused. Grounding and rephrasing.")
                rephrased = ground_and_rephrase(
                    client=openai_client,
                    model=model,
                    prev_robot_response=last_robot_response,
                    user_confused_utterance=user_utt,
                    history=history,
                )
                speak(furhat, rephrased, chosen_gesture)
                history.append({"role": "assistant", "content": rephrased})
                last_robot_response = rephrased
                continue

            if chosen_label.strip().lower() == "other-language":
                print("User spoke another language.")
                fallback = "Sorry, I only understand English. Could you repeat that in English?"
                speak(furhat, fallback, chosen_gesture)
                history.append({"role": "assistant", "content": fallback})
                last_robot_response = fallback
                continue

            # Normal routing: exit the inner loop to process state transition
            pending_gesture = chosen_gesture
            break

        next_edge = graph.find_edge_by_label(node, chosen_label)
        if not next_edge:
            print(f"No matching edge for label '{chosen_label}' from node '{node.title}'")
            break

        current_node = graph.get_node(next_edge.target)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Furhat robot IP address")
    parser.add_argument("--auth_key", type=str, default="admin", help="Authentication key for Realtime API")
    parser.add_argument(
        "--dialogue",
        type=str,
        default=None,
        help="Optional explicit path to a dialogue JSON file. Overrides DIALOGUE_FILE_SELECTION.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OPENAI_MODEL,
        help="OpenAI model name",
    )
    args = parser.parse_args()

    load_dotenv(override=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment/.env")
        sys.exit(1)

    openai_client = OpenAI(api_key=api_key)

    furhat = FurhatClient(host=args.host, auth_key=args.auth_key)
    furhat.set_logging_level(logging.INFO)

    try:
        furhat.connect()
    except Exception:
        print(f"Failed to connect to Furhat on {args.host}.")
        sys.exit(1)

    try:
        dialogue_path = args.dialogue or resolve_dialogue_file(
            DIALOGUE_DIR,
            DIALOGUE_FILE_SELECTION,
        )
        print(f"Using dialogue file: {dialogue_path}")
        graph = DialogueGraph.from_file(dialogue_path)
    except Exception as e:
        print(f"Failed to load dialogue file: {e}")
        sys.exit(1)

    furhat.request_voice_config(name="Neural", gender="Neutral", language="en-US") 

    #face_status = furhat.request_face_status()
    #print(face_status["face_list"])   # available face names/ids
    furhat.request_face_config("adult - Jane", True, True)

    furhat.request_face_headpose(0, 0.1, 0, False)

    furhat.request_gesture_start("BigSmile")

    run_dialogue(
        graph=graph,
        openai_client=openai_client,
        furhat=furhat,
        model=args.model,
    )