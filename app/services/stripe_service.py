"""Stripe billing — checkout/portal sessions + webhook event handling.

`handle_event` is a pure function (event dict -> subscription update) so the
webhook logic is unit-testable without Stripe. The Stripe SDK is lazy-imported
only for live calls (checkout/portal/signature verification).

We carry `user_id` and `plan` in Stripe metadata so webhooks map back to our user
without an extra customer lookup.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.config import Settings
from app.services.repo import Repo

# plan -> Stripe price id comes from settings (env); kept here for clarity.
PLAN_PRICE_ENV = {"starter": "STRIPE_PRICE_STARTER", "pro": "STRIPE_PRICE_PRO"}


def handle_event(event_type: str, obj: Dict[str, Any], repo: Repo) -> Optional[Dict[str, Any]]:
    """Apply a (already verified) Stripe event to our subscription state.

    Returns the updated subscription row, or None if the event is ignored.
    """
    meta = obj.get("metadata", {}) or {}
    user_id = meta.get("user_id")
    if not user_id:
        return None

    if event_type == "checkout.session.completed":
        return repo.set_subscription(
            user_id, plan=meta.get("plan", "starter"), status="active",
            stripe_subscription_id=obj.get("subscription"),
        )
    if event_type == "customer.subscription.updated":
        return repo.set_subscription(
            user_id, status=obj.get("status", "active"),
            stripe_subscription_id=obj.get("id"),
            current_period_end=obj.get("current_period_end"),
        )
    if event_type == "customer.subscription.deleted":
        return repo.set_subscription(user_id, plan="free", status="canceled")
    return None


def create_checkout_session(user: Dict[str, Any], plan: str, settings: Settings,
                            success_url: str, cancel_url: str) -> str:
    """Create a Stripe Checkout session; return its URL. Lazy-imports stripe."""
    import os

    import stripe  # lazy
    stripe.api_key = settings.stripe_secret_key
    price_env = PLAN_PRICE_ENV.get(plan)
    price_id = os.getenv(price_env) if price_env else None
    if not price_id:
        raise ValueError(f"no Stripe price configured for plan {plan!r}")
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url, cancel_url=cancel_url,
        customer_email=user.get("email"),
        metadata={"user_id": user["id"], "plan": plan},
        subscription_data={"metadata": {"user_id": user["id"], "plan": plan}},
    )
    return session.url


def create_portal_session(customer_id: str, settings: Settings, return_url: str) -> str:
    """Create a Stripe billing-portal session; return its URL."""
    import stripe  # lazy
    stripe.api_key = settings.stripe_secret_key
    return stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url).url


def verify_webhook(payload: bytes, sig_header: str, settings: Settings) -> Dict[str, Any]:
    """Verify a webhook signature and return the event. Lazy-imports stripe."""
    import stripe  # lazy
    return stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
