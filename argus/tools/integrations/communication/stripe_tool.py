"""
Stripe Tool for ARGUS.

Payment processing and subscription management.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class StripeTool(BaseTool):
    """
    Stripe - Payment processing and subscription management.
    
    Features:
    - Customer management
    - Payment intents
    - Subscriptions
    - Invoices
    - Products and prices
    - Payment methods
    
    Example:
        >>> tool = StripeTool()
        >>> result = tool(action="create_customer", email="user@example.com")
        >>> result = tool(action="create_payment_intent", amount=1000, currency="usd")
    """
    
    name = "stripe"
    description = "Payment processing and subscription management"
    category = ToolCategory.FINANCE
    version = "1.0.0"
    
    BASE_URL = "https://api.stripe.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("STRIPE_API_KEY")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Stripe API key provided")
        
        logger.debug("Stripe tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.auth = (self.api_key, "")
            self._session.headers.update({
                "Content-Type": "application/x-www-form-urlencoded",
            })
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Make API request to Stripe."""
        session = self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            data=data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error = error_data.get("error", {})
                raise Exception(error.get("message", f"HTTP {response.status_code}"))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_customers",
        customer_id: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        amount: Optional[int] = None,
        currency: str = "usd",
        payment_method_id: Optional[str] = None,
        payment_intent_id: Optional[str] = None,
        product_id: Optional[str] = None,
        price_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        invoice_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Stripe operations."""
        actions = {
            # Customer operations
            "list_customers": self._list_customers,
            "get_customer": self._get_customer,
            "create_customer": self._create_customer,
            "update_customer": self._update_customer,
            "delete_customer": self._delete_customer,
            
            # Payment Intent operations
            "list_payment_intents": self._list_payment_intents,
            "get_payment_intent": self._get_payment_intent,
            "create_payment_intent": self._create_payment_intent,
            "confirm_payment_intent": self._confirm_payment_intent,
            "cancel_payment_intent": self._cancel_payment_intent,
            
            # Subscription operations
            "list_subscriptions": self._list_subscriptions,
            "get_subscription": self._get_subscription,
            "create_subscription": self._create_subscription,
            "update_subscription": self._update_subscription,
            "cancel_subscription": self._cancel_subscription,
            
            # Invoice operations
            "list_invoices": self._list_invoices,
            "get_invoice": self._get_invoice,
            "create_invoice": self._create_invoice,
            "finalize_invoice": self._finalize_invoice,
            "pay_invoice": self._pay_invoice,
            
            # Product operations
            "list_products": self._list_products,
            "get_product": self._get_product,
            "create_product": self._create_product,
            
            # Price operations
            "list_prices": self._list_prices,
            "get_price": self._get_price,
            "create_price": self._create_price,
            
            # Balance
            "get_balance": self._get_balance,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("Stripe API key not configured")
            
            return actions[action](
                customer_id=customer_id,
                email=email,
                name=name,
                description=description,
                amount=amount,
                currency=currency,
                payment_method_id=payment_method_id,
                payment_intent_id=payment_intent_id,
                product_id=product_id,
                price_id=price_id,
                subscription_id=subscription_id,
                invoice_id=invoice_id,
                metadata=metadata,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Stripe error: {e}")
            return ToolResult.from_error(f"Stripe error: {e}")
    
    # Customer operations
    def _list_customers(self, limit: int = 100, **kwargs) -> ToolResult:
        """List customers."""
        response = self._request(
            "GET",
            "/customers",
            params={"limit": min(limit, 100)},
        )
        
        customers = [
            {
                "id": c["id"],
                "email": c.get("email"),
                "name": c.get("name"),
                "created": c.get("created"),
            }
            for c in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "customers": customers,
            "count": len(customers),
            "has_more": response.get("has_more", False),
        })
    
    def _get_customer(
        self,
        customer_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get customer details."""
        if not customer_id:
            return ToolResult.from_error("customer_id is required")
        
        response = self._request("GET", f"/customers/{customer_id}")
        
        return ToolResult.from_data({
            "customer": {
                "id": response.get("id"),
                "email": response.get("email"),
                "name": response.get("name"),
                "description": response.get("description"),
                "balance": response.get("balance"),
                "created": response.get("created"),
                "metadata": response.get("metadata"),
            }
        })
    
    def _create_customer(
        self,
        email: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a customer."""
        data = {}
        
        if email:
            data["email"] = email
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = v
        
        response = self._request("POST", "/customers", data=data)
        
        return ToolResult.from_data({
            "customer_id": response.get("id"),
            "email": response.get("email"),
            "created": True,
        })
    
    def _update_customer(
        self,
        customer_id: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a customer."""
        if not customer_id:
            return ToolResult.from_error("customer_id is required")
        
        data = {}
        if email:
            data["email"] = email
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        
        if not data:
            return ToolResult.from_error("No fields to update")
        
        response = self._request("POST", f"/customers/{customer_id}", data=data)
        
        return ToolResult.from_data({
            "customer_id": response.get("id"),
            "updated": True,
        })
    
    def _delete_customer(
        self,
        customer_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a customer."""
        if not customer_id:
            return ToolResult.from_error("customer_id is required")
        
        response = self._request("DELETE", f"/customers/{customer_id}")
        
        return ToolResult.from_data({
            "customer_id": customer_id,
            "deleted": response.get("deleted", True),
        })
    
    # Payment Intent operations
    def _list_payment_intents(self, limit: int = 100, **kwargs) -> ToolResult:
        """List payment intents."""
        params = {"limit": min(limit, 100)}
        
        customer_id = kwargs.get("customer_id")
        if customer_id:
            params["customer"] = customer_id
        
        response = self._request("GET", "/payment_intents", params=params)
        
        intents = [
            {
                "id": pi["id"],
                "amount": pi.get("amount"),
                "currency": pi.get("currency"),
                "status": pi.get("status"),
                "customer": pi.get("customer"),
                "created": pi.get("created"),
            }
            for pi in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "payment_intents": intents,
            "count": len(intents),
        })
    
    def _get_payment_intent(
        self,
        payment_intent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get payment intent details."""
        if not payment_intent_id:
            return ToolResult.from_error("payment_intent_id is required")
        
        response = self._request("GET", f"/payment_intents/{payment_intent_id}")
        
        return ToolResult.from_data({
            "payment_intent": {
                "id": response.get("id"),
                "amount": response.get("amount"),
                "currency": response.get("currency"),
                "status": response.get("status"),
                "customer": response.get("customer"),
                "payment_method": response.get("payment_method"),
                "client_secret": response.get("client_secret"),
            }
        })
    
    def _create_payment_intent(
        self,
        amount: Optional[int] = None,
        currency: str = "usd",
        customer_id: Optional[str] = None,
        payment_method_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a payment intent."""
        if not amount:
            return ToolResult.from_error("amount is required (in cents)")
        
        data = {
            "amount": amount,
            "currency": currency,
        }
        
        if customer_id:
            data["customer"] = customer_id
        if payment_method_id:
            data["payment_method"] = payment_method_id
        if metadata:
            for k, v in metadata.items():
                data[f"metadata[{k}]"] = v
        
        response = self._request("POST", "/payment_intents", data=data)
        
        return ToolResult.from_data({
            "payment_intent_id": response.get("id"),
            "client_secret": response.get("client_secret"),
            "status": response.get("status"),
            "created": True,
        })
    
    def _confirm_payment_intent(
        self,
        payment_intent_id: Optional[str] = None,
        payment_method_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Confirm a payment intent."""
        if not payment_intent_id:
            return ToolResult.from_error("payment_intent_id is required")
        
        data = {}
        if payment_method_id:
            data["payment_method"] = payment_method_id
        
        response = self._request(
            "POST",
            f"/payment_intents/{payment_intent_id}/confirm",
            data=data,
        )
        
        return ToolResult.from_data({
            "payment_intent_id": response.get("id"),
            "status": response.get("status"),
            "confirmed": True,
        })
    
    def _cancel_payment_intent(
        self,
        payment_intent_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Cancel a payment intent."""
        if not payment_intent_id:
            return ToolResult.from_error("payment_intent_id is required")
        
        response = self._request(
            "POST",
            f"/payment_intents/{payment_intent_id}/cancel",
        )
        
        return ToolResult.from_data({
            "payment_intent_id": response.get("id"),
            "status": response.get("status"),
            "canceled": True,
        })
    
    # Subscription operations
    def _list_subscriptions(
        self,
        customer_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List subscriptions."""
        params = {"limit": min(limit, 100)}
        
        if customer_id:
            params["customer"] = customer_id
        
        response = self._request("GET", "/subscriptions", params=params)
        
        subscriptions = [
            {
                "id": s["id"],
                "customer": s.get("customer"),
                "status": s.get("status"),
                "current_period_end": s.get("current_period_end"),
                "cancel_at_period_end": s.get("cancel_at_period_end"),
            }
            for s in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "subscriptions": subscriptions,
            "count": len(subscriptions),
        })
    
    def _get_subscription(
        self,
        subscription_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get subscription details."""
        if not subscription_id:
            return ToolResult.from_error("subscription_id is required")
        
        response = self._request("GET", f"/subscriptions/{subscription_id}")
        
        return ToolResult.from_data({
            "subscription": {
                "id": response.get("id"),
                "customer": response.get("customer"),
                "status": response.get("status"),
                "current_period_start": response.get("current_period_start"),
                "current_period_end": response.get("current_period_end"),
                "items": response.get("items", {}).get("data", []),
            }
        })
    
    def _create_subscription(
        self,
        customer_id: Optional[str] = None,
        price_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a subscription."""
        if not customer_id:
            return ToolResult.from_error("customer_id is required")
        if not price_id:
            return ToolResult.from_error("price_id is required")
        
        data = {
            "customer": customer_id,
            "items[0][price]": price_id,
        }
        
        response = self._request("POST", "/subscriptions", data=data)
        
        return ToolResult.from_data({
            "subscription_id": response.get("id"),
            "status": response.get("status"),
            "created": True,
        })
    
    def _update_subscription(
        self,
        subscription_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a subscription."""
        if not subscription_id:
            return ToolResult.from_error("subscription_id is required")
        
        data = {}
        
        cancel_at_period_end = kwargs.get("cancel_at_period_end")
        if cancel_at_period_end is not None:
            data["cancel_at_period_end"] = str(cancel_at_period_end).lower()
        
        if not data:
            return ToolResult.from_error("No fields to update")
        
        response = self._request(
            "POST",
            f"/subscriptions/{subscription_id}",
            data=data,
        )
        
        return ToolResult.from_data({
            "subscription_id": response.get("id"),
            "status": response.get("status"),
            "updated": True,
        })
    
    def _cancel_subscription(
        self,
        subscription_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Cancel a subscription."""
        if not subscription_id:
            return ToolResult.from_error("subscription_id is required")
        
        response = self._request("DELETE", f"/subscriptions/{subscription_id}")
        
        return ToolResult.from_data({
            "subscription_id": response.get("id"),
            "status": response.get("status"),
            "canceled": True,
        })
    
    # Invoice operations
    def _list_invoices(
        self,
        customer_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List invoices."""
        params = {"limit": min(limit, 100)}
        
        if customer_id:
            params["customer"] = customer_id
        
        response = self._request("GET", "/invoices", params=params)
        
        invoices = [
            {
                "id": i["id"],
                "customer": i.get("customer"),
                "status": i.get("status"),
                "amount_due": i.get("amount_due"),
                "amount_paid": i.get("amount_paid"),
                "currency": i.get("currency"),
            }
            for i in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "invoices": invoices,
            "count": len(invoices),
        })
    
    def _get_invoice(
        self,
        invoice_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get invoice details."""
        if not invoice_id:
            return ToolResult.from_error("invoice_id is required")
        
        response = self._request("GET", f"/invoices/{invoice_id}")
        
        return ToolResult.from_data({"invoice": response})
    
    def _create_invoice(
        self,
        customer_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create an invoice."""
        if not customer_id:
            return ToolResult.from_error("customer_id is required")
        
        data = {"customer": customer_id}
        
        response = self._request("POST", "/invoices", data=data)
        
        return ToolResult.from_data({
            "invoice_id": response.get("id"),
            "status": response.get("status"),
            "created": True,
        })
    
    def _finalize_invoice(
        self,
        invoice_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Finalize an invoice."""
        if not invoice_id:
            return ToolResult.from_error("invoice_id is required")
        
        response = self._request("POST", f"/invoices/{invoice_id}/finalize")
        
        return ToolResult.from_data({
            "invoice_id": response.get("id"),
            "status": response.get("status"),
            "finalized": True,
        })
    
    def _pay_invoice(
        self,
        invoice_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Pay an invoice."""
        if not invoice_id:
            return ToolResult.from_error("invoice_id is required")
        
        response = self._request("POST", f"/invoices/{invoice_id}/pay")
        
        return ToolResult.from_data({
            "invoice_id": response.get("id"),
            "status": response.get("status"),
            "paid": True,
        })
    
    # Product operations
    def _list_products(self, limit: int = 100, **kwargs) -> ToolResult:
        """List products."""
        response = self._request(
            "GET",
            "/products",
            params={"limit": min(limit, 100)},
        )
        
        products = [
            {
                "id": p["id"],
                "name": p.get("name"),
                "description": p.get("description"),
                "active": p.get("active"),
            }
            for p in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "products": products,
            "count": len(products),
        })
    
    def _get_product(
        self,
        product_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get product details."""
        if not product_id:
            return ToolResult.from_error("product_id is required")
        
        response = self._request("GET", f"/products/{product_id}")
        
        return ToolResult.from_data({"product": response})
    
    def _create_product(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a product."""
        if not name:
            return ToolResult.from_error("name is required")
        
        data = {"name": name}
        
        if description:
            data["description"] = description
        
        response = self._request("POST", "/products", data=data)
        
        return ToolResult.from_data({
            "product_id": response.get("id"),
            "name": response.get("name"),
            "created": True,
        })
    
    # Price operations
    def _list_prices(
        self,
        product_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List prices."""
        params = {"limit": min(limit, 100)}
        
        if product_id:
            params["product"] = product_id
        
        response = self._request("GET", "/prices", params=params)
        
        prices = [
            {
                "id": p["id"],
                "product": p.get("product"),
                "unit_amount": p.get("unit_amount"),
                "currency": p.get("currency"),
                "type": p.get("type"),
                "active": p.get("active"),
            }
            for p in response.get("data", [])
        ]
        
        return ToolResult.from_data({
            "prices": prices,
            "count": len(prices),
        })
    
    def _get_price(
        self,
        price_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get price details."""
        if not price_id:
            return ToolResult.from_error("price_id is required")
        
        response = self._request("GET", f"/prices/{price_id}")
        
        return ToolResult.from_data({"price": response})
    
    def _create_price(
        self,
        product_id: Optional[str] = None,
        amount: Optional[int] = None,
        currency: str = "usd",
        **kwargs,
    ) -> ToolResult:
        """Create a price."""
        if not product_id:
            return ToolResult.from_error("product_id is required")
        if not amount:
            return ToolResult.from_error("amount is required (in cents)")
        
        data = {
            "product": product_id,
            "unit_amount": amount,
            "currency": currency,
        }
        
        recurring = kwargs.get("recurring")
        if recurring:
            data["recurring[interval]"] = recurring
        
        response = self._request("POST", "/prices", data=data)
        
        return ToolResult.from_data({
            "price_id": response.get("id"),
            "created": True,
        })
    
    # Balance
    def _get_balance(self, **kwargs) -> ToolResult:
        """Get account balance."""
        response = self._request("GET", "/balance")
        
        available = [
            {"amount": b["amount"], "currency": b["currency"]}
            for b in response.get("available", [])
        ]
        
        pending = [
            {"amount": b["amount"], "currency": b["currency"]}
            for b in response.get("pending", [])
        ]
        
        return ToolResult.from_data({
            "available": available,
            "pending": pending,
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
                            "list_customers", "get_customer", "create_customer",
                            "update_customer", "delete_customer",
                            "list_payment_intents", "get_payment_intent",
                            "create_payment_intent", "confirm_payment_intent",
                            "cancel_payment_intent",
                            "list_subscriptions", "get_subscription",
                            "create_subscription", "update_subscription",
                            "cancel_subscription",
                            "list_invoices", "get_invoice", "create_invoice",
                            "finalize_invoice", "pay_invoice",
                            "list_products", "get_product", "create_product",
                            "list_prices", "get_price", "create_price",
                            "get_balance",
                        ],
                    },
                    "customer_id": {"type": "string"},
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                    "amount": {"type": "integer", "description": "Amount in cents"},
                    "currency": {"type": "string", "default": "usd"},
                    "payment_intent_id": {"type": "string"},
                    "payment_method_id": {"type": "string"},
                    "subscription_id": {"type": "string"},
                    "product_id": {"type": "string"},
                    "price_id": {"type": "string"},
                    "invoice_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
