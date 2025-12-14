"""
Lightweight tester for post_sentry SIP outbound flow.

Usage:
    python scripts/test_post_sentry.py --to +33753823988

Requires env:
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_SIP_TRUNK_ID
    (optional) LIVEKIT_SIP_TO for default destination.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _add_repo_paths() -> None:
    """
    Add the LeKiwi-runtime root to sys.path whether this script lives in
    the repo root or inside LeKiwi-runtime/scripts.
    """
    here = Path(__file__).resolve()
    parent = here.parents[1]  # If script is at repo/scripts or runtime/scripts
    # If already inside LeKiwi-runtime, use it directly; otherwise append it.
    if (parent / "post_sentry.py").exists():
        runtime_root = parent
    elif (parent / "LeKiwi-runtime").exists():
        runtime_root = parent / "LeKiwi-runtime"
    else:
        runtime_root = parent  # fallback
    sys.path.append(str(runtime_root))


_add_repo_paths()

from post_sentry import (  # noqa: E402
    create_sip_participant,
    _normalize_base_url,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test SIP outbound from post_sentry.")
    parser.add_argument("--to", dest="to_number", help="Destination E.164 number")
    parser.add_argument(
        "--room-name",
        default="sentry-alert-room",
        help="LiveKit room name to bridge the call into",
    )
    parser.add_argument(
        "--participant-identity",
        default="sip-alert",
        help="Identity for the SIP participant",
    )
    parser.add_argument(
        "--participant-name",
        default="Sentry Alert",
        help="Display name for the SIP participant",
    )
    parser.add_argument(
        "--wait",
        dest="wait_until_answered",
        action="store_true",
        help="Wait until call is answered before returning",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait_until_answered",
        action="store_false",
        help="Return immediately after dialing",
    )
    parser.set_defaults(wait_until_answered=True)

    args = parser.parse_args()

    livekit_url = _normalize_base_url(os.getenv("LIVEKIT_URL"))
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    bearer_token = os.getenv("TOKEN") or os.getenv("LIVEKIT_BEARER_TOKEN")
    sip_trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID")
    destination = args.to_number or os.getenv("LIVEKIT_SIP_TO")

    missing = [
        name
        for name, val in [
            ("LIVEKIT_URL", livekit_url),
            ("LIVEKIT_API_KEY", api_key if not bearer_token else "ok"),
            ("LIVEKIT_API_SECRET", api_secret if not bearer_token else "ok"),
            ("LIVEKIT_SIP_TRUNK_ID", sip_trunk_id),
            ("destination", destination),
        ]
        if not val
    ]
    if missing:
        print(f"Missing required config: {', '.join(missing)}")
        return 1

    payload = {
        "sip_trunk_id": sip_trunk_id,
        "sip_call_to": destination,
        "room_name": args.room_name,
        "participant_identity": args.participant_identity,
        "participant_name": args.participant_name,
        "wait_until_answered": bool(args.wait_until_answered),
    }

    if bearer_token:
        print("Auth: Bearer (TOKEN/LIVEKIT_BEARER_TOKEN present)")
    else:
        print("Auth: Basic (no TOKEN found)")

    ok, result = await create_sip_participant(
        base_url=livekit_url,
        api_key=api_key or "",  # type: ignore[arg-type]
        api_secret=api_secret or "",  # type: ignore[arg-type]
        payload=payload,
        bearer_token=bearer_token,
    )
    print("success" if ok else "failed", result)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
