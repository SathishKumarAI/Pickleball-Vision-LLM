"""Billing — Stripe checkout, portal, and webhook."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.deps import get_current_user, get_repo
from app.services import stripe_service
from app.services.repo import Repo

router = APIRouter(prefix="/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    plan: str  # starter | pro
    success_url: str
    cancel_url: str


class PortalRequest(BaseModel):
    return_url: str


@router.post("/checkout")
def checkout(req: CheckoutRequest, user=Depends(get_current_user),
             settings: Settings = Depends(get_settings)):
    try:
        url = stripe_service.create_checkout_session(
            user, req.plan, settings, req.success_url, req.cancel_url)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    return {"url": url}


@router.post("/portal")
def portal(req: PortalRequest, user=Depends(get_current_user),
           repo: Repo = Depends(get_repo), settings: Settings = Depends(get_settings)):
    sub = repo.get_subscription(user["id"])
    customer = sub.get("stripe_customer_id")
    if not customer:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no Stripe customer")
    return {"url": stripe_service.create_portal_session(customer, settings, req.return_url)}


@router.post("/webhook")
async def webhook(request: Request, stripe_signature: str = Header(None),
                  repo: Repo = Depends(get_repo), settings: Settings = Depends(get_settings)):
    payload = await request.body()
    try:
        event = stripe_service.verify_webhook(payload, stripe_signature, settings)
    except Exception as e:  # noqa: BLE001 - bad signature / parse
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"invalid webhook: {e}") from e
    stripe_service.handle_event(event["type"], event["data"]["object"], repo)
    return {"received": True}
