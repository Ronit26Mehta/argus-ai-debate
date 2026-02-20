"""
PayPal Tool for ARGUS.

Payment processing and order management.
"""

from __future__ import annotations

import os
import base64
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class PayPalTool(BaseTool):
    """
    PayPal - Payment processing and order management.
    
    Features:
    - Order creation and capture
    - Payment processing
    - Refunds
    - Payout management
    - Subscription management
    
    Example:
        >>> tool = PayPalTool()
        >>> result = tool(action="create_order", amount="10.00", currency="USD")
        >>> result = tool(action="capture_order", order_id="ORDER123")
    """
    
    name = "paypal"
    description = "Payment processing and order management"
    category = ToolCategory.FINANCE
    version = "1.0.0"
    
    SANDBOX_URL = "https://api-m.sandbox.paypal.com"
    LIVE_URL = "https://api-m.paypal.com"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        sandbox: bool = True,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.client_id = client_id or os.getenv("PAYPAL_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PAYPAL_CLIENT_SECRET")
        self.sandbox = sandbox if sandbox is not None else os.getenv("PAYPAL_SANDBOX", "true").lower() == "true"
        
        self.base_url = self.SANDBOX_URL if self.sandbox else self.LIVE_URL
        
        self._session = None
        self._access_token = None
        
        if not self.client_id or not self.client_secret:
            logger.warning("PayPal credentials not fully provided")
        
        logger.debug(f"PayPal tool initialized (sandbox={self.sandbox})")
    
    def _get_access_token(self) -> str:
        """Get OAuth access token."""
        if self._access_token:
            return self._access_token
        
        import requests
        
        auth_str = f"{self.client_id}:{self.client_secret}"
        auth_bytes = base64.b64encode(auth_str.encode()).decode()
        
        response = requests.post(
            f"{self.base_url}/v1/oauth2/token",
            headers={
                "Authorization": f"Basic {auth_bytes}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        
        if response.status_code >= 400:
            raise Exception(f"Failed to get access token: {response.text}")
        
        data = response.json()
        self._access_token = data["access_token"]
        return self._access_token
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            
            token = self._get_access_token()
            self._session.headers.update({
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            })
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Make API request to PayPal."""
        session = self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            json=data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("message", "")
                details = error_data.get("details", [])
                if details:
                    message += ": " + "; ".join(d.get("description", "") for d in details)
                raise Exception(message or f"HTTP {response.status_code}")
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_orders",
        order_id: Optional[str] = None,
        capture_id: Optional[str] = None,
        payment_id: Optional[str] = None,
        payout_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        amount: Optional[str] = None,
        currency: str = "USD",
        description: Optional[str] = None,
        return_url: Optional[str] = None,
        cancel_url: Optional[str] = None,
        items: Optional[list] = None,
        recipient_email: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute PayPal operations."""
        actions = {
            # Order operations
            "create_order": self._create_order,
            "get_order": self._get_order,
            "capture_order": self._capture_order,
            "authorize_order": self._authorize_order,
            
            # Capture operations
            "refund_capture": self._refund_capture,
            "get_capture": self._get_capture,
            
            # Payout operations
            "create_payout": self._create_payout,
            "get_payout": self._get_payout,
            
            # Subscription operations
            "list_plans": self._list_plans,
            "create_plan": self._create_plan,
            "get_plan": self._get_plan,
            "create_subscription": self._create_subscription,
            "get_subscription": self._get_subscription,
            "cancel_subscription": self._cancel_subscription,
            
            # Product operations
            "list_products": self._list_products,
            "create_product": self._create_product,
            "get_product": self._get_product,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.client_id or not self.client_secret:
                return ToolResult.from_error("PayPal credentials not configured")
            
            return actions[action](
                order_id=order_id,
                capture_id=capture_id,
                payment_id=payment_id,
                payout_id=payout_id,
                subscription_id=subscription_id,
                plan_id=plan_id,
                amount=amount,
                currency=currency,
                description=description,
                return_url=return_url,
                cancel_url=cancel_url,
                items=items,
                recipient_email=recipient_email,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"PayPal error: {e}")
            return ToolResult.from_error(f"PayPal error: {e}")
    
    # Order operations
    def _create_order(
        self,
        amount: Optional[str] = None,
        currency: str = "USD",
        description: Optional[str] = None,
        return_url: Optional[str] = None,
        cancel_url: Optional[str] = None,
        items: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Create an order."""
        if not amount:
            return ToolResult.from_error("amount is required")
        
        purchase_unit = {
            "amount": {
                "currency_code": currency,
                "value": amount,
            },
        }
        
        if description:
            purchase_unit["description"] = description
        
        if items:
            item_total = sum(float(i.get("unit_amount", 0)) * int(i.get("quantity", 1)) for i in items)
            purchase_unit["amount"]["breakdown"] = {
                "item_total": {
                    "currency_code": currency,
                    "value": f"{item_total:.2f}",
                }
            }
            purchase_unit["items"] = [
                {
                    "name": item.get("name", "Item"),
                    "quantity": str(item.get("quantity", 1)),
                    "unit_amount": {
                        "currency_code": currency,
                        "value": str(item.get("unit_amount", "0.00")),
                    },
                }
                for item in items
            ]
        
        data = {
            "intent": "CAPTURE",
            "purchase_units": [purchase_unit],
        }
        
        if return_url and cancel_url:
            data["application_context"] = {
                "return_url": return_url,
                "cancel_url": cancel_url,
            }
        
        response = self._request("POST", "/v2/checkout/orders", data=data)
        
        # Extract approval URL
        approval_url = None
        for link in response.get("links", []):
            if link.get("rel") == "approve":
                approval_url = link.get("href")
                break
        
        return ToolResult.from_data({
            "order_id": response.get("id"),
            "status": response.get("status"),
            "approval_url": approval_url,
            "created": True,
        })
    
    def _get_order(
        self,
        order_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get order details."""
        if not order_id:
            return ToolResult.from_error("order_id is required")
        
        response = self._request("GET", f"/v2/checkout/orders/{order_id}")
        
        return ToolResult.from_data({
            "order": {
                "id": response.get("id"),
                "status": response.get("status"),
                "intent": response.get("intent"),
                "purchase_units": response.get("purchase_units"),
                "create_time": response.get("create_time"),
            }
        })
    
    def _capture_order(
        self,
        order_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Capture payment for an order."""
        if not order_id:
            return ToolResult.from_error("order_id is required")
        
        response = self._request("POST", f"/v2/checkout/orders/{order_id}/capture")
        
        # Extract capture ID
        capture_id = None
        captures = response.get("purchase_units", [{}])[0].get("payments", {}).get("captures", [])
        if captures:
            capture_id = captures[0].get("id")
        
        return ToolResult.from_data({
            "order_id": response.get("id"),
            "status": response.get("status"),
            "capture_id": capture_id,
            "captured": True,
        })
    
    def _authorize_order(
        self,
        order_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Authorize payment for an order."""
        if not order_id:
            return ToolResult.from_error("order_id is required")
        
        response = self._request("POST", f"/v2/checkout/orders/{order_id}/authorize")
        
        return ToolResult.from_data({
            "order_id": response.get("id"),
            "status": response.get("status"),
            "authorized": True,
        })
    
    # Capture operations
    def _refund_capture(
        self,
        capture_id: Optional[str] = None,
        amount: Optional[str] = None,
        currency: str = "USD",
        **kwargs,
    ) -> ToolResult:
        """Refund a captured payment."""
        if not capture_id:
            return ToolResult.from_error("capture_id is required")
        
        data = {}
        if amount:
            data["amount"] = {
                "value": amount,
                "currency_code": currency,
            }
        
        response = self._request(
            "POST",
            f"/v2/payments/captures/{capture_id}/refund",
            data=data if data else None,
        )
        
        return ToolResult.from_data({
            "refund_id": response.get("id"),
            "status": response.get("status"),
            "refunded": True,
        })
    
    def _get_capture(
        self,
        capture_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get capture details."""
        if not capture_id:
            return ToolResult.from_error("capture_id is required")
        
        response = self._request("GET", f"/v2/payments/captures/{capture_id}")
        
        return ToolResult.from_data({"capture": response})
    
    # Payout operations
    def _create_payout(
        self,
        recipient_email: Optional[str] = None,
        amount: Optional[str] = None,
        currency: str = "USD",
        **kwargs,
    ) -> ToolResult:
        """Create a payout."""
        if not recipient_email:
            return ToolResult.from_error("recipient_email is required")
        if not amount:
            return ToolResult.from_error("amount is required")
        
        import uuid
        
        data = {
            "sender_batch_header": {
                "sender_batch_id": str(uuid.uuid4())[:12],
                "email_subject": kwargs.get("subject", "You have a payout!"),
            },
            "items": [
                {
                    "recipient_type": "EMAIL",
                    "amount": {
                        "value": amount,
                        "currency": currency,
                    },
                    "receiver": recipient_email,
                    "note": kwargs.get("note", ""),
                }
            ],
        }
        
        response = self._request("POST", "/v1/payments/payouts", data=data)
        
        batch_header = response.get("batch_header", {})
        
        return ToolResult.from_data({
            "payout_batch_id": batch_header.get("payout_batch_id"),
            "batch_status": batch_header.get("batch_status"),
            "created": True,
        })
    
    def _get_payout(
        self,
        payout_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get payout details."""
        if not payout_id:
            return ToolResult.from_error("payout_id is required")
        
        response = self._request("GET", f"/v1/payments/payouts/{payout_id}")
        
        batch_header = response.get("batch_header", {})
        
        return ToolResult.from_data({
            "payout_batch_id": batch_header.get("payout_batch_id"),
            "batch_status": batch_header.get("batch_status"),
            "items": response.get("items", []),
        })
    
    # Product operations
    def _list_products(self, **kwargs) -> ToolResult:
        """List products."""
        response = self._request("GET", "/v1/catalogs/products")
        
        products = [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "description": p.get("description"),
                "type": p.get("type"),
            }
            for p in response.get("products", [])
        ]
        
        return ToolResult.from_data({
            "products": products,
            "count": len(products),
        })
    
    def _create_product(
        self,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a product."""
        name = kwargs.get("name")
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {
            "name": name,
            "type": kwargs.get("type", "SERVICE"),
        }
        
        if description:
            data["description"] = description
        
        response = self._request("POST", "/v1/catalogs/products", data=data)
        
        return ToolResult.from_data({
            "product_id": response.get("id"),
            "name": response.get("name"),
            "created": True,
        })
    
    def _get_product(self, **kwargs) -> ToolResult:
        """Get product details."""
        product_id = kwargs.get("product_id")
        if not product_id:
            return ToolResult.from_error("product_id is required")
        
        response = self._request("GET", f"/v1/catalogs/products/{product_id}")
        
        return ToolResult.from_data({"product": response})
    
    # Plan operations
    def _list_plans(self, **kwargs) -> ToolResult:
        """List subscription plans."""
        response = self._request("GET", "/v1/billing/plans")
        
        plans = [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "status": p.get("status"),
                "product_id": p.get("product_id"),
            }
            for p in response.get("plans", [])
        ]
        
        return ToolResult.from_data({
            "plans": plans,
            "count": len(plans),
        })
    
    def _create_plan(
        self,
        amount: Optional[str] = None,
        currency: str = "USD",
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a subscription plan."""
        product_id = kwargs.get("product_id")
        name = kwargs.get("name")
        
        if not product_id:
            return ToolResult.from_error("product_id is required")
        if not name:
            return ToolResult.from_error("name is required")
        if not amount:
            return ToolResult.from_error("amount is required")
        
        interval = kwargs.get("interval", "MONTH")
        
        data = {
            "product_id": product_id,
            "name": name,
            "billing_cycles": [
                {
                    "frequency": {
                        "interval_unit": interval,
                        "interval_count": 1,
                    },
                    "tenure_type": "REGULAR",
                    "sequence": 1,
                    "total_cycles": 0,
                    "pricing_scheme": {
                        "fixed_price": {
                            "value": amount,
                            "currency_code": currency,
                        }
                    },
                }
            ],
            "payment_preferences": {
                "auto_bill_outstanding": True,
                "payment_failure_threshold": 3,
            },
        }
        
        if description:
            data["description"] = description
        
        response = self._request("POST", "/v1/billing/plans", data=data)
        
        return ToolResult.from_data({
            "plan_id": response.get("id"),
            "status": response.get("status"),
            "created": True,
        })
    
    def _get_plan(
        self,
        plan_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get plan details."""
        if not plan_id:
            return ToolResult.from_error("plan_id is required")
        
        response = self._request("GET", f"/v1/billing/plans/{plan_id}")
        
        return ToolResult.from_data({"plan": response})
    
    # Subscription operations
    def _create_subscription(
        self,
        plan_id: Optional[str] = None,
        return_url: Optional[str] = None,
        cancel_url: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a subscription."""
        if not plan_id:
            return ToolResult.from_error("plan_id is required")
        
        data = {"plan_id": plan_id}
        
        if return_url and cancel_url:
            data["application_context"] = {
                "return_url": return_url,
                "cancel_url": cancel_url,
            }
        
        response = self._request("POST", "/v1/billing/subscriptions", data=data)
        
        # Extract approval URL
        approval_url = None
        for link in response.get("links", []):
            if link.get("rel") == "approve":
                approval_url = link.get("href")
                break
        
        return ToolResult.from_data({
            "subscription_id": response.get("id"),
            "status": response.get("status"),
            "approval_url": approval_url,
            "created": True,
        })
    
    def _get_subscription(
        self,
        subscription_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get subscription details."""
        if not subscription_id:
            return ToolResult.from_error("subscription_id is required")
        
        response = self._request("GET", f"/v1/billing/subscriptions/{subscription_id}")
        
        return ToolResult.from_data({
            "subscription": {
                "id": response.get("id"),
                "status": response.get("status"),
                "plan_id": response.get("plan_id"),
                "start_time": response.get("start_time"),
                "billing_info": response.get("billing_info"),
            }
        })
    
    def _cancel_subscription(
        self,
        subscription_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Cancel a subscription."""
        if not subscription_id:
            return ToolResult.from_error("subscription_id is required")
        
        reason = kwargs.get("reason", "Cancellation requested")
        
        self._request(
            "POST",
            f"/v1/billing/subscriptions/{subscription_id}/cancel",
            data={"reason": reason},
        )
        
        return ToolResult.from_data({
            "subscription_id": subscription_id,
            "canceled": True,
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "create_order", "get_order", "capture_order", "authorize_order",
                            "refund_capture", "get_capture",
                            "create_payout", "get_payout",
                            "list_products", "create_product", "get_product",
                            "list_plans", "create_plan", "get_plan",
                            "create_subscription", "get_subscription", "cancel_subscription",
                        ],
                    },
                    "order_id": {"type": "string"},
                    "capture_id": {"type": "string"},
                    "payout_id": {"type": "string"},
                    "subscription_id": {"type": "string"},
                    "plan_id": {"type": "string"},
                    "amount": {"type": "string", "description": "Amount as string (e.g., '10.00')"},
                    "currency": {"type": "string", "default": "USD"},
                    "description": {"type": "string"},
                    "return_url": {"type": "string"},
                    "cancel_url": {"type": "string"},
                    "recipient_email": {"type": "string"},
                    "items": {"type": "array"},
                },
                "required": ["action"],
            },
        }
