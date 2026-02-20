"""
OpenAPI REST Integration Module for ARGUS.

Dynamic REST API client generation from OpenAPI/Swagger specifications.
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Optional, Any, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlencode
from functools import lru_cache

logger = logging.getLogger(__name__)


class AuthType(Enum):
    """Authentication types supported by OpenAPI."""
    NONE = "none"
    API_KEY = "apiKey"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"


@dataclass
class SecurityScheme:
    """Security scheme configuration."""
    type: AuthType
    name: str = ""
    location: str = "header"  # header, query, cookie
    scheme: str = ""  # bearer, basic
    bearer_format: str = ""
    flows: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIParameter:
    """API parameter definition."""
    name: str
    location: str  # path, query, header, cookie
    required: bool = False
    schema: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    default: Any = None
    
    @property
    def param_type(self) -> str:
        return self.schema.get("type", "string")


@dataclass
class APIOperation:
    """API operation definition."""
    operation_id: str
    method: str
    path: str
    summary: str = ""
    description: str = ""
    parameters: List[APIParameter] = field(default_factory=list)
    request_body: Dict[str, Any] = field(default_factory=dict)
    responses: Dict[str, Any] = field(default_factory=dict)
    security: List[Dict[str, List[str]]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass  
class APISpec:
    """Parsed OpenAPI specification."""
    title: str
    version: str
    description: str = ""
    base_url: str = ""
    servers: List[Dict[str, Any]] = field(default_factory=list)
    operations: Dict[str, APIOperation] = field(default_factory=dict)
    security_schemes: Dict[str, SecurityScheme] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)


class OpenAPIParser:
    """Parser for OpenAPI/Swagger specifications."""
    
    def __init__(self):
        self._ref_cache: Dict[str, Any] = {}
    
    def parse(self, spec: Dict[str, Any]) -> APISpec:
        """Parse OpenAPI specification."""
        openapi_version = spec.get("openapi", spec.get("swagger", "2.0"))
        is_v3 = openapi_version.startswith("3")
        
        info = spec.get("info", {})
        title = info.get("title", "API")
        version = info.get("version", "1.0.0")
        description = info.get("description", "")
        
        # Parse servers/base URL
        servers = []
        base_url = ""
        
        if is_v3:
            servers = spec.get("servers", [])
            if servers:
                base_url = servers[0].get("url", "")
        else:
            host = spec.get("host", "")
            base_path = spec.get("basePath", "")
            schemes = spec.get("schemes", ["https"])
            if host:
                base_url = f"{schemes[0]}://{host}{base_path}"
        
        # Parse security schemes
        security_schemes = {}
        
        if is_v3:
            components = spec.get("components", {})
            sec_schemes = components.get("securitySchemes", {})
        else:
            sec_schemes = spec.get("securityDefinitions", {})
        
        for name, scheme_def in sec_schemes.items():
            security_schemes[name] = self._parse_security_scheme(name, scheme_def)
        
        # Store components for reference resolution
        self._ref_cache = spec.get("components", spec.get("definitions", {}))
        
        # Parse operations
        operations = {}
        paths = spec.get("paths", {})
        
        for path, path_item in paths.items():
            # Handle path-level parameters
            path_params = self._parse_parameters(path_item.get("parameters", []))
            
            for method in ["get", "post", "put", "patch", "delete", "head", "options"]:
                if method not in path_item:
                    continue
                
                op_def = path_item[method]
                operation = self._parse_operation(path, method, op_def, path_params, is_v3)
                operations[operation.operation_id] = operation
        
        return APISpec(
            title=title,
            version=version,
            description=description,
            base_url=base_url,
            servers=servers,
            operations=operations,
            security_schemes=security_schemes,
            components=self._ref_cache,
        )
    
    def _parse_security_scheme(self, name: str, scheme_def: Dict[str, Any]) -> SecurityScheme:
        """Parse a security scheme definition."""
        scheme_type = scheme_def.get("type", "")
        
        if scheme_type == "apiKey":
            return SecurityScheme(
                type=AuthType.API_KEY,
                name=scheme_def.get("name", ""),
                location=scheme_def.get("in", "header"),
            )
        elif scheme_type == "http":
            scheme = scheme_def.get("scheme", "bearer").lower()
            if scheme == "bearer":
                return SecurityScheme(
                    type=AuthType.BEARER,
                    scheme="bearer",
                    bearer_format=scheme_def.get("bearerFormat", ""),
                )
            elif scheme == "basic":
                return SecurityScheme(type=AuthType.BASIC, scheme="basic")
        elif scheme_type == "oauth2":
            return SecurityScheme(
                type=AuthType.OAUTH2,
                flows=scheme_def.get("flows", scheme_def.get("flow", {})),
            )
        
        return SecurityScheme(type=AuthType.NONE)
    
    def _parse_parameters(self, params: List[Dict[str, Any]]) -> List[APIParameter]:
        """Parse parameter definitions."""
        result = []
        
        for param in params:
            # Resolve references
            if "$ref" in param:
                param = self._resolve_ref(param["$ref"])
            
            result.append(APIParameter(
                name=param.get("name", ""),
                location=param.get("in", "query"),
                required=param.get("required", False),
                schema=param.get("schema", param),
                description=param.get("description", ""),
                default=param.get("default"),
            ))
        
        return result
    
    def _parse_operation(
        self,
        path: str,
        method: str,
        op_def: Dict[str, Any],
        path_params: List[APIParameter],
        is_v3: bool,
    ) -> APIOperation:
        """Parse an operation definition."""
        # Generate operation ID if not provided
        operation_id = op_def.get("operationId")
        if not operation_id:
            # Generate from path and method
            clean_path = re.sub(r"[{}]", "", path)
            clean_path = re.sub(r"[^a-zA-Z0-9]", "_", clean_path)
            operation_id = f"{method}_{clean_path}".strip("_")
        
        # Merge path-level and operation-level parameters
        op_params = self._parse_parameters(op_def.get("parameters", []))
        
        # Combine, with operation params taking precedence
        param_map = {p.name: p for p in path_params}
        for p in op_params:
            param_map[p.name] = p
        
        parameters = list(param_map.values())
        
        # Parse request body
        request_body = {}
        if is_v3 and "requestBody" in op_def:
            request_body = op_def["requestBody"]
            if "$ref" in request_body:
                request_body = self._resolve_ref(request_body["$ref"])
        elif not is_v3:
            # v2 body parameters
            for param in op_def.get("parameters", []):
                if param.get("in") == "body":
                    request_body = {
                        "content": {
                            "application/json": {
                                "schema": param.get("schema", {})
                            }
                        },
                        "required": param.get("required", False),
                    }
                    break
        
        return APIOperation(
            operation_id=operation_id,
            method=method.upper(),
            path=path,
            summary=op_def.get("summary", ""),
            description=op_def.get("description", ""),
            parameters=parameters,
            request_body=request_body,
            responses=op_def.get("responses", {}),
            security=op_def.get("security", []),
            tags=op_def.get("tags", []),
            deprecated=op_def.get("deprecated", False),
        )
    
    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        """Resolve a $ref reference."""
        if ref.startswith("#/"):
            parts = ref[2:].split("/")
            current = {"components": self._ref_cache, "definitions": self._ref_cache}
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return {}
            
            return current if isinstance(current, dict) else {}
        
        return {}


class OpenAPIClient:
    """
    Dynamic REST API client generated from OpenAPI specification.
    
    Example:
        >>> client = OpenAPIClient.from_url("https://api.example.com/openapi.json")
        >>> result = client.execute("getUsers", limit=10)
        >>> 
        >>> # Or call operation directly
        >>> result = client.get_users(limit=10)
    """
    
    def __init__(
        self,
        spec: APISpec,
        base_url: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ):
        self.spec = spec
        self.base_url = (base_url or spec.base_url).rstrip("/")
        self.auth_config = auth_config or {}
        self.timeout = timeout
        
        self._session = None
        self._setup_methods()
        
        logger.debug(f"OpenAPI client initialized for {spec.title} v{spec.version}")
    
    @classmethod
    def from_dict(
        cls,
        spec_dict: Dict[str, Any],
        base_url: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> "OpenAPIClient":
        """Create client from specification dictionary."""
        parser = OpenAPIParser()
        spec = parser.parse(spec_dict)
        return cls(spec, base_url, auth_config, timeout)
    
    @classmethod
    def from_json(
        cls,
        json_str: str,
        base_url: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> "OpenAPIClient":
        """Create client from JSON string."""
        spec_dict = json.loads(json_str)
        return cls.from_dict(spec_dict, base_url, auth_config, timeout)
    
    @classmethod
    def from_file(
        cls,
        file_path: str,
        base_url: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> "OpenAPIClient":
        """Create client from specification file."""
        with open(file_path, "r") as f:
            content = f.read()
        
        if file_path.endswith((".yaml", ".yml")):
            try:
                import yaml
                spec_dict = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for YAML specs: pip install pyyaml")
        else:
            spec_dict = json.loads(content)
        
        return cls.from_dict(spec_dict, base_url, auth_config, timeout)
    
    @classmethod
    def from_url(
        cls,
        url: str,
        base_url: Optional[str] = None,
        auth_config: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> "OpenAPIClient":
        """Create client from specification URL."""
        import requests
        
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "")
        
        if "yaml" in content_type or url.endswith((".yaml", ".yml")):
            try:
                import yaml
                spec_dict = yaml.safe_load(response.text)
            except ImportError:
                raise ImportError("PyYAML required for YAML specs: pip install pyyaml")
        else:
            spec_dict = response.json()
        
        return cls.from_dict(spec_dict, base_url, auth_config, timeout)
    
    def _get_session(self):
        """Get HTTP session with authentication configured."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            
            # Configure authentication
            self._configure_auth()
        
        return self._session
    
    def _configure_auth(self):
        """Configure session authentication."""
        if not self.auth_config:
            return
        
        session = self._session
        
        for scheme_name, credentials in self.auth_config.items():
            scheme = self.spec.security_schemes.get(scheme_name)
            
            if not scheme:
                continue
            
            if scheme.type == AuthType.API_KEY:
                if scheme.location == "header":
                    session.headers[scheme.name] = credentials
                elif scheme.location == "query":
                    # Handle in request
                    pass
            
            elif scheme.type == AuthType.BEARER:
                token = credentials if isinstance(credentials, str) else credentials.get("token", "")
                session.headers["Authorization"] = f"Bearer {token}"
            
            elif scheme.type == AuthType.BASIC:
                if isinstance(credentials, dict):
                    username = credentials.get("username", "")
                    password = credentials.get("password", "")
                    session.auth = (username, password)
                elif isinstance(credentials, tuple):
                    session.auth = credentials
    
    def _setup_methods(self):
        """Setup dynamic methods for each operation."""
        for op_id, operation in self.spec.operations.items():
            # Convert operation ID to snake_case method name
            method_name = self._to_snake_case(op_id)
            
            # Create method
            method = self._create_method(operation)
            
            # Bind to instance
            setattr(self, method_name, method)
    
    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase/PascalCase to snake_case."""
        # Insert underscore before uppercase letters
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    
    def _create_method(self, operation: APIOperation) -> Callable:
        """Create a method for an operation."""
        def method(**kwargs):
            return self.execute(operation.operation_id, **kwargs)
        
        # Set docstring
        doc_parts = []
        if operation.summary:
            doc_parts.append(operation.summary)
        if operation.description:
            doc_parts.append(operation.description)
        if operation.parameters:
            doc_parts.append("\nParameters:")
            for param in operation.parameters:
                required = " (required)" if param.required else ""
                doc_parts.append(f"    {param.name}: {param.param_type}{required}")
                if param.description:
                    doc_parts.append(f"        {param.description}")
        
        method.__doc__ = "\n".join(doc_parts) if doc_parts else None
        method.__name__ = self._to_snake_case(operation.operation_id)
        
        return method
    
    def execute(self, operation_id: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an API operation.
        
        Args:
            operation_id: The operation ID from the OpenAPI spec
            **kwargs: Operation parameters
        
        Returns:
            Response data as dictionary
        """
        operation = self.spec.operations.get(operation_id)
        
        if not operation:
            raise ValueError(f"Unknown operation: {operation_id}")
        
        return self._execute_operation(operation, **kwargs)
    
    def _execute_operation(self, operation: APIOperation, **kwargs) -> Dict[str, Any]:
        """Execute an operation."""
        session = self._get_session()
        
        # Build URL with path parameters
        url = self.base_url + operation.path
        
        query_params = {}
        headers = {}
        body = None
        
        # Process parameters
        for param in operation.parameters:
            value = kwargs.pop(param.name, param.default)
            
            if value is None:
                if param.required:
                    raise ValueError(f"Required parameter missing: {param.name}")
                continue
            
            if param.location == "path":
                url = url.replace(f"{{{param.name}}}", str(value))
            elif param.location == "query":
                query_params[param.name] = value
            elif param.location == "header":
                headers[param.name] = str(value)
        
        # Handle request body
        if operation.request_body:
            body_data = kwargs.pop("body", kwargs.pop("data", None))
            
            if body_data is None and kwargs:
                # Use remaining kwargs as body
                body_data = kwargs
            
            if body_data:
                content = operation.request_body.get("content", {})
                
                if "application/json" in content:
                    headers["Content-Type"] = "application/json"
                    body = json.dumps(body_data)
                elif "multipart/form-data" in content:
                    # Handle file uploads separately
                    body = body_data
                else:
                    body = body_data
        
        # Build final URL with query params
        if query_params:
            url = f"{url}?{urlencode(query_params)}"
        
        # Make request
        response = session.request(
            method=operation.method,
            url=url,
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        
        # Handle response
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        
        try:
            result["data"] = response.json()
        except ValueError:
            result["data"] = response.text
        
        if response.status_code >= 400:
            result["error"] = True
            result["message"] = str(result["data"])
        else:
            result["error"] = False
        
        return result
    
    def list_operations(self) -> List[Dict[str, Any]]:
        """List all available operations."""
        operations = []
        
        for op_id, operation in self.spec.operations.items():
            operations.append({
                "operation_id": op_id,
                "method": operation.method,
                "path": operation.path,
                "summary": operation.summary,
                "tags": operation.tags,
                "deprecated": operation.deprecated,
            })
        
        return operations
    
    def get_operation_schema(self, operation_id: str) -> Dict[str, Any]:
        """Get detailed schema for an operation."""
        operation = self.spec.operations.get(operation_id)
        
        if not operation:
            raise ValueError(f"Unknown operation: {operation_id}")
        
        params = []
        for param in operation.parameters:
            params.append({
                "name": param.name,
                "in": param.location,
                "required": param.required,
                "type": param.param_type,
                "description": param.description,
                "default": param.default,
            })
        
        return {
            "operation_id": operation_id,
            "method": operation.method,
            "path": operation.path,
            "summary": operation.summary,
            "description": operation.description,
            "parameters": params,
            "request_body": operation.request_body,
            "responses": operation.responses,
            "security": operation.security,
            "tags": operation.tags,
            "deprecated": operation.deprecated,
        }


class OpenAPIToolGenerator:
    """
    Generate ARGUS tools from OpenAPI specifications.
    
    Creates BaseTool subclasses for API operations.
    """
    
    def __init__(self, spec: APISpec):
        self.spec = spec
        self.parser = OpenAPIParser()
    
    @classmethod
    def from_url(cls, url: str) -> "OpenAPIToolGenerator":
        """Create generator from specification URL."""
        import requests
        
        response = requests.get(url)
        response.raise_for_status()
        
        parser = OpenAPIParser()
        spec = parser.parse(response.json())
        
        return cls(spec)
    
    @classmethod
    def from_file(cls, file_path: str) -> "OpenAPIToolGenerator":
        """Create generator from specification file."""
        with open(file_path, "r") as f:
            content = f.read()
        
        if file_path.endswith((".yaml", ".yml")):
            import yaml
            spec_dict = yaml.safe_load(content)
        else:
            spec_dict = json.loads(content)
        
        parser = OpenAPIParser()
        spec = parser.parse(spec_dict)
        
        return cls(spec)
    
    def generate_tool_class(
        self,
        class_name: Optional[str] = None,
        operations: Optional[List[str]] = None,
    ) -> str:
        """
        Generate Python source code for a BaseTool subclass.
        
        Args:
            class_name: Name for the generated class
            operations: List of operation IDs to include (all if None)
        
        Returns:
            Python source code string
        """
        if not class_name:
            # Generate from API title
            clean_title = re.sub(r"[^a-zA-Z0-9]", "", self.spec.title)
            class_name = f"{clean_title}Tool"
        
        # Filter operations
        ops = self.spec.operations
        if operations:
            ops = {k: v for k, v in ops.items() if k in operations}
        
        # Generate code
        lines = [
            '"""',
            f"Auto-generated tool for {self.spec.title} API.",
            "",
            f"Version: {self.spec.version}",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "import os",
            "import logging",
            "from typing import Optional, Any",
            "",
            "from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory",
            "",
            "logger = logging.getLogger(__name__)",
            "",
            "",
            f"class {class_name}(BaseTool):",
            f'    """',
            f"    {self.spec.title} API Tool.",
            f"    ",
            f"    {self.spec.description}" if self.spec.description else "",
            f'    """',
            "",
            f'    name = "{self._to_snake_case(class_name.replace("Tool", ""))}"',
            f'    description = "{self.spec.description[:100] if self.spec.description else self.spec.title}"',
            "    category = ToolCategory.AUTOMATION",
            f'    version = "{self.spec.version}"',
            "",
            "    def __init__(",
            "        self,",
            "        base_url: Optional[str] = None,",
            "        api_key: Optional[str] = None,",
            "        config: Optional[ToolConfig] = None,",
            "    ):",
            "        super().__init__(config)",
            "",
            f'        self.base_url = (base_url or os.getenv("{self._to_env_var(class_name)}_URL", "{self.spec.base_url}")).rstrip("/")',
            f'        self.api_key = api_key or os.getenv("{self._to_env_var(class_name)}_API_KEY")',
            "",
            "        self._session = None",
            "",
            "    def _get_session(self):",
            '        """Get HTTP session."""',
            "        if self._session is None:",
            "            import requests",
            "            self._session = requests.Session()",
            '            headers = {"Content-Type": "application/json"}',
            "            if self.api_key:",
            '                headers["Authorization"] = f"Bearer {self.api_key}"',
            "            self._session.headers.update(headers)",
            "        return self._session",
            "",
        ]
        
        # Generate execute method
        action_enum = [f'"{op_id}"' for op_id in ops.keys()]
        
        lines.extend([
            "    def execute(",
            "        self,",
            '        action: str = "' + (list(ops.keys())[0] if ops else "") + '",',
            "        **kwargs: Any,",
            "    ) -> ToolResult:",
            '        """Execute API operation."""',
            "        actions = {",
        ])
        
        for op_id in ops.keys():
            method_name = f"_action_{self._to_snake_case(op_id)}"
            lines.append(f'            "{op_id}": self.{method_name},')
        
        lines.extend([
            "        }",
            "",
            "        if action not in actions:",
            "            return ToolResult.from_error(",
            '                f"Unknown action: {action}. Available: {list(actions.keys())}"',
            "            )",
            "",
            "        try:",
            "            return actions[action](**kwargs)",
            "        except Exception as e:",
            f'            logger.error(f"{class_name} error: {{e}}")',
            f'            return ToolResult.from_error(f"{class_name} error: {{e}}")',
            "",
        ])
        
        # Generate action methods
        for op_id, operation in ops.items():
            lines.extend(self._generate_action_method(operation))
        
        # Generate schema method
        lines.extend([
            "    def get_schema(self) -> dict[str, Any]:",
            "        return {",
            "            **super().get_schema(),",
            '            "parameters": {',
            '                "type": "object",',
            '                "properties": {',
            '                    "action": {',
            '                        "type": "string",',
            f'                        "enum": [{", ".join(action_enum)}],',
            "                    },",
            "                },",
            '                "required": ["action"],',
            "            },",
            "        }",
        ])
        
        return "\n".join(lines)
    
    def _generate_action_method(self, operation: APIOperation) -> List[str]:
        """Generate an action method for an operation."""
        method_name = f"_action_{self._to_snake_case(operation.operation_id)}"
        
        lines = [
            f"    def {method_name}(",
            "        self,",
            "        **kwargs,",
            "    ) -> ToolResult:",
            f'        """',
            f"        {operation.summary}" if operation.summary else f"        {operation.operation_id}",
            f"        ",
            f"        {operation.method} {operation.path}",
            f'        """',
        ]
        
        # Build URL
        path = operation.path
        path_params = [p for p in operation.parameters if p.location == "path"]
        query_params = [p for p in operation.parameters if p.location == "query"]
        
        # Extract required path params
        for param in path_params:
            lines.append(f"        {param.name} = kwargs.get('{param.name}')")
            if param.required:
                lines.append(f"        if not {param.name}:")
                lines.append(f"            return ToolResult.from_error('{param.name} is required')")
        
        # Build URL string
        if path_params:
            url_parts = []
            for param in path_params:
                path = path.replace(f"{{{param.name}}}", f"{{{param.name}}}")
            lines.append(f'        url = f"{{self.base_url}}{path}"')
        else:
            lines.append(f'        url = f"{{self.base_url}}{path}"')
        
        # Handle query params
        if query_params:
            lines.append("        params = {}")
            for param in query_params:
                lines.append(f"        if kwargs.get('{param.name}') is not None:")
                lines.append(f"            params['{param.name}'] = kwargs['{param.name}']")
        else:
            lines.append("        params = None")
        
        # Handle body
        if operation.request_body:
            lines.extend([
                "        data = kwargs.get('data', kwargs.get('body'))",
                "        if data is None:",
                "            data = {k: v for k, v in kwargs.items() if not k.startswith('_')}",
            ])
        else:
            lines.append("        data = None")
        
        # Make request
        lines.extend([
            "",
            "        session = self._get_session()",
            f'        response = session.request("{operation.method}", url, params=params, json=data)',
            "",
            "        if response.status_code >= 400:",
            "            try:",
            "                error_data = response.json()",
            '                message = error_data.get("message", error_data.get("error", f"HTTP {response.status_code}"))',
            "                return ToolResult.from_error(str(message))",
            "            except ValueError:",
            '                return ToolResult.from_error(f"HTTP {response.status_code}")',
            "",
            "        try:",
            '            return ToolResult.from_data({"data": response.json()})',
            "        except ValueError:",
            '            return ToolResult.from_data({"data": response.text})',
            "",
        ])
        
        return lines
    
    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    
    def _to_env_var(self, name: str) -> str:
        """Convert to environment variable format."""
        return self._to_snake_case(name.replace("Tool", "")).upper()


# Utility functions

def load_openapi_spec(source: str) -> APISpec:
    """
    Load an OpenAPI specification from various sources.
    
    Args:
        source: URL, file path, or JSON string
    
    Returns:
        Parsed APISpec object
    """
    parser = OpenAPIParser()
    
    if source.startswith(("http://", "https://")):
        import requests
        response = requests.get(source)
        response.raise_for_status()
        spec_dict = response.json()
    elif os.path.isfile(source):
        with open(source, "r") as f:
            content = f.read()
        if source.endswith((".yaml", ".yml")):
            import yaml
            spec_dict = yaml.safe_load(content)
        else:
            spec_dict = json.loads(content)
    else:
        spec_dict = json.loads(source)
    
    return parser.parse(spec_dict)


def create_client(
    source: str,
    base_url: Optional[str] = None,
    auth_config: Optional[Dict[str, Any]] = None,
) -> OpenAPIClient:
    """
    Create an OpenAPI client from various sources.
    
    Args:
        source: URL, file path, or JSON string
        base_url: Override base URL
        auth_config: Authentication configuration
    
    Returns:
        Configured OpenAPIClient
    """
    spec = load_openapi_spec(source)
    return OpenAPIClient(spec, base_url, auth_config)


def generate_tool_code(
    source: str,
    class_name: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """
    Generate tool code from OpenAPI specification.
    
    Args:
        source: URL, file path, or JSON string
        class_name: Name for the generated class
        output_file: Optional file path to write code to
    
    Returns:
        Generated Python source code
    """
    spec = load_openapi_spec(source)
    generator = OpenAPIToolGenerator(spec)
    
    code = generator.generate_tool_class(class_name)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(code)
    
    return code


__all__ = [
    "AuthType",
    "SecurityScheme",
    "APIParameter",
    "APIOperation",
    "APISpec",
    "OpenAPIParser",
    "OpenAPIClient",
    "OpenAPIToolGenerator",
    "load_openapi_spec",
    "create_client",
    "generate_tool_code",
]
