#!/usr/bin/env python3
"""
Better Playwright MCP Server

A Model Context Protocol (MCP) server that provides comprehensive browser automation
capabilities using Playwright. This server enables AI assistants to interact with web
pages through actions like navigation, element interaction, form filling, and page
analysis.

Features:
- Browser lifecycle management with CDP connection support
- Element interaction (click, type, extract text/links)
- Navigation and history management
- Form input handling with multiple selector types
- Page snapshots (accessibility tree and visual screenshots)
- Automatic browser cleanup and error handling

The server can connect to existing browser instances via Chrome DevTools Protocol (CDP)
or launch new browser instances as needed.
"""

import json
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from pydantic import BaseModel, Field
from typing import Literal


@dataclass
class AppContext:
    browser: Browser
    context: BrowserContext
    page: Page


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage browser lifecycle"""
    # Initialize browser on startup
    playwright = await async_playwright().start()
    browser = None

    # Check for CDP URL in environment
    cdp_url = os.environ.get("LOCAL_CDP_URL", "http://localhost:9222")

    try:
        # Try to connect to existing browser via CDP
        browser = await playwright.chromium.connect_over_cdp(cdp_url)
        print(f"Connected to existing browser at {cdp_url}")
    except Exception as e:
        print(f"Failed to connect to CDP at {cdp_url}: {e}")
        # Fall back to launching new browser
        browser = await playwright.chromium.launch(
            headless=False,  # Set to True for headless mode
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        print("Launched new browser instance")

    context = await browser.new_context()
    page = await context.new_page()

    try:
        yield AppContext(browser=browser, context=context, page=page)
    finally:
        # Cleanup on shutdown
        await context.close()
        await browser.close()
        await playwright.stop()


mcp = FastMCP(
    "better-playwright",
    description="Browser automation server using Playwright for web interaction, navigation, and page analysis",
    lifespan=app_lifespan,
)


class ElementActionInput(BaseModel):
    type: Literal["className", "id", "text"] = Field(
        ...,
        description="How to select the element - 'className' for CSS class names, 'id' for element IDs, 'text' for visible text content",
    )
    selector: str = Field(
        ...,
        description="The selector value: class name (without dot), element ID (without hash), or exact visible text content",
    )
    action: Literal["click", "getText", "extractLinks", "getRawElement", "type"] = (
        Field(
            ...,
            description="Action to perform: 'click' to click element, 'getText' to get text content, 'extractLinks' to get all links within element, 'getRawElement' to get element details, 'type' to input text",
        )
    )
    text: str | None = Field(
        None,
        description="Text to type into the element (required only when action is 'type')",
    )


class NavigateInput(BaseModel):
    type: Literal["url", "back", "forward", "refresh"] = Field(
        "url",
        description="Type of navigation: 'url' to navigate to a specific URL, 'back' to go back in history, 'forward' to go forward in history, 'refresh' to reload current page",
    )
    url: str | None = Field(
        None,
        description="The URL to navigate to (required only when type is 'url'). Should include protocol (http:// or https://)",
    )


class SnapshotInput(BaseModel):
    type: Literal["accessibility", "image", "accessibility_summary", "accessibility_scoped"] = Field(
        ...,
        description="Type of snapshot: 'accessibility' for full tree (use sparingly), 'accessibility_summary' for just interactive elements, 'accessibility_scoped' for specific element subtree, 'image' for visual screenshot",
    )
    selector_type: Literal["className", "id", "text", "role"] | None = Field(
        None,
        description="For accessibility_scoped: how to select the root element - 'className', 'id', 'text', or 'role'",
    )
    selector: str | None = Field(
        None,
        description="For accessibility_scoped: the selector value to scope the accessibility tree to a specific element",
    )


class FillInputInput(BaseModel):
    type: Literal["className", "id", "text", "placeholder", "label"] = Field(
        ...,
        description="How to select the input element: 'className' for CSS class, 'id' for element ID, 'text' for visible text, 'placeholder' for placeholder text, 'label' for associated label text",
    )
    selector: str = Field(
        ...,
        description="The selector value: class name (without dot), element ID (without hash), visible text content, placeholder text, or label text",
    )
    value: str = Field(
        ...,
        description="The text to fill into the input field. Will replace any existing content",
    )


class ExploreByRoleInput(BaseModel):
    role: Literal["banner", "main", "navigation", "contentinfo", "form", "region", "article", "section", "complementary", "search"] = Field(
        ...,
        description="ARIA landmark role to explore: 'banner' (header), 'main' (main content), 'navigation' (nav menus), 'contentinfo' (footer), 'form', 'region', 'article', 'section', 'complementary' (sidebar), 'search'",
    )


@mcp.tool()
def get_active_page(ctx) -> Page:
    """Get the currently active page instance for internal use by other tools"""
    app_context = ctx.request_context.lifespan_context
    return app_context.page


@mcp.tool()
async def getElement(input: ElementActionInput) -> str:
    """Find an element on the page and perform actions like clicking, getting text, or typing. Use this for interacting with buttons, links, text content, and form elements. Supports selecting elements by CSS class, ID, or visible text content."""
    ctx = mcp.get_context()
    page = get_active_page(ctx)

    # Build selector based on type
    if input.type == "className":
        selector = f".{input.selector}"
    elif input.type == "id":
        selector = f"#{input.selector}"
    elif input.type == "text":
        selector = f"text={input.selector}"
    else:
        return (
            f"Error: Invalid type '{input.type}'. Must be 'className', 'id', or 'text'"
        )

    try:
        element = await page.query_selector(selector)
        if not element:
            return f"Element not found with selector: {selector}"

        # Perform the requested action
        if input.action == "click":
            await element.click()
            return f"Successfully clicked element with selector: {selector}"

        elif input.action == "getText":
            text_content = await element.text_content()
            return json.dumps(
                {"action": "getText", "selector": selector, "text": text_content},
                indent=2,
            )

        elif input.action == "extractLinks":
            # Get all links within this element
            links = await element.query_selector_all("a")
            link_data = []
            for link in links:
                href = await link.get_attribute("href")
                text = await link.text_content()
                link_data.append({"href": href, "text": text})

            return json.dumps(
                {"action": "extractLinks", "selector": selector, "links": link_data},
                indent=2,
            )

        elif input.action == "getRawElement":
            element_info = {
                "action": "getRawElement",
                "selector": selector,
                "tag_name": await element.evaluate("el => el.tagName.toLowerCase()"),
                "text_content": await element.text_content(),
                "visible": await element.is_visible(),
                "enabled": await element.is_enabled(),
                "attributes": await element.evaluate(
                    "el => Object.fromEntries([...el.attributes].map(attr => [attr.name, attr.value]))"
                ),
            }
            return json.dumps(element_info, indent=2)

        elif input.action == "type":
            if input.text is None:
                return "Error: 'text' parameter is required when action is 'type'"

            # Clear existing text and type new text
            await element.clear()
            await element.type(input.text)
            return f"Successfully typed '{input.text}' into element with selector: {selector}"

        else:
            return f"Error: Invalid action '{input.action}'. Must be 'click', 'getText', 'extractLinks', 'getRawElement', or 'type'"

    except Exception as e:
        return f"Error performing action '{input.action}' on element: {str(e)}"


@mcp.tool()
async def navigate(input: NavigateInput) -> str:
    """Navigate to a specific URL or perform browser history navigation. Use this to visit websites, go back/forward in browser history, or refresh the current page. Essential for browsing between different web pages."""
    ctx = mcp.get_context()
    page = get_active_page(ctx)

    try:
        if input.type == "url":
            if input.url is None:
                return "Error: url parameter is required when type is 'url'"
            await page.goto(input.url)
            return f"Successfully navigated to {input.url}"

        elif input.type == "back":
            await page.go_back()
            return "Successfully navigated back"

        elif input.type == "forward":
            await page.go_forward()
            return "Successfully navigated forward"

        elif input.type == "refresh":
            await page.reload()
            return "Successfully refreshed the page"

        else:
            return f"Error: Invalid navigation type '{input.type}'. Must be 'url', 'back', 'forward', or 'refresh'"

    except Exception as e:
        return f"Error performing navigation '{input.type}': {str(e)}"


def _build_selector(selector_type: str, selector: str) -> str:
    """Build a playwright selector from type and value"""
    if selector_type == "className":
        return f".{selector}"
    elif selector_type == "id":
        return f"#{selector}"
    elif selector_type == "text":
        return f"text={selector}"
    elif selector_type == "role":
        return f"role={selector}"
    else:
        raise ValueError(f"Invalid selector type: {selector_type}")


def _filter_interactive_elements(node, max_depth=3, current_depth=0):
    """Filter accessibility tree to show only interactive elements, with depth limit"""
    if not node or current_depth > max_depth:
        return None
    
    # Interactive roles we care about
    interactive_roles = {
        'button', 'link', 'textbox', 'combobox', 'listbox', 'option', 
        'checkbox', 'radio', 'tab', 'menuitem', 'treeitem', 'gridcell',
        'columnheader', 'rowheader', 'searchbox', 'slider', 'spinbutton'
    }
    
    # Always include certain structural roles at any depth
    structural_roles = {'main', 'navigation', 'banner', 'contentinfo', 'form', 'region'}
    
    result = None
    filtered_children = []
    
    # Process children first
    for child in node.get('children', []):
        filtered_child = _filter_interactive_elements(child, max_depth, current_depth + 1)
        if filtered_child:
            filtered_children.append(filtered_child)
    
    # Include this node if it's interactive, structural, or has interesting children
    role = node.get('role', '').lower()
    if (role in interactive_roles or 
        role in structural_roles or 
        filtered_children or
        current_depth == 0):  # Always include root
        
        result = {
            'role': node.get('role'),
            'name': node.get('name'),
        }
        
        # Add key attributes for interactive elements
        if role in interactive_roles:
            for attr in ['value', 'checked', 'selected', 'expanded', 'disabled', 'level']:
                if attr in node:
                    result[attr] = node[attr]
        
        if filtered_children:
            result['children'] = filtered_children
    
    return result


@mcp.tool()
async def getSnapshot(input: SnapshotInput) -> str:
    """Capture page state as accessibility tree or screenshot. Use 'accessibility_summary' for interactive elements overview, 'accessibility_scoped' to explore specific sections, 'accessibility' for full tree (avoid for large pages), 'image' for visual confirmation."""
    ctx = mcp.get_context()
    page = get_active_page(ctx)

    try:
        if input.type == "accessibility":
            # Get full accessibility tree snapshot (use sparingly)
            accessibility_tree = await page.accessibility.snapshot()
            return json.dumps(
                {"type": "accessibility", "snapshot": accessibility_tree}, indent=2
            )

        elif input.type == "accessibility_summary":
            # Get filtered tree showing only interactive elements
            accessibility_tree = await page.accessibility.snapshot()
            if accessibility_tree:
                summary = _filter_interactive_elements(accessibility_tree)
                return json.dumps(
                    {"type": "accessibility_summary", "snapshot": summary}, indent=2
                )
            else:
                return json.dumps({"type": "accessibility_summary", "snapshot": None}, indent=2)

        elif input.type == "accessibility_scoped":
            # Get accessibility tree scoped to specific element
            if not input.selector_type or not input.selector:
                return "Error: selector_type and selector required for accessibility_scoped"
            
            try:
                selector = _build_selector(input.selector_type, input.selector)
                element = await page.query_selector(selector)
                if not element:
                    return f"Error: Element not found with selector: {selector}"
                
                # Get scoped accessibility tree
                scoped_tree = await page.accessibility.snapshot(root=element)
                return json.dumps(
                    {
                        "type": "accessibility_scoped", 
                        "selector": selector,
                        "snapshot": scoped_tree
                    }, 
                    indent=2
                )
            except Exception as e:
                return f"Error getting scoped accessibility tree: {str(e)}"

        elif input.type == "image":
            # Take a screenshot and return base64 encoded image
            screenshot_bytes = await page.screenshot(
                full_page=True, type="jpeg", quality=50
            )
            import base64

            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            return json.dumps(
                {
                    "type": "image",
                    "format": "jpeg",
                    "data": screenshot_base64,
                    "encoding": "base64",
                },
                indent=2,
            )

    except Exception as e:
        return f"Error capturing {input.type} snapshot: {str(e)}"


@mcp.tool()
async def fillInput(input: FillInputInput) -> str:
    """Fill text into form input fields like search boxes, text areas, and input forms. This is the preferred method for entering text into form elements as it properly handles input events and validation. Can target inputs by CSS class, ID, placeholder text, associated labels, or visible text."""
    ctx = mcp.get_context()
    page = get_active_page(ctx)

    # Build selector based on type
    if input.type == "className":
        selector = f".{input.selector}"
    elif input.type == "id":
        selector = f"#{input.selector}"
    elif input.type == "text":
        selector = f"text={input.selector}"
    elif input.type == "placeholder":
        selector = f"[placeholder='{input.selector}']"
    elif input.type == "label":
        selector = f"label:text('{input.selector}') >> input"
    else:
        return f"Error: Invalid type '{input.type}'. Must be 'className', 'id', 'text', 'placeholder', or 'label'"

    try:
        # Use Playwright's fill method for input fields
        await page.fill(selector, input.value)
        return f"Successfully filled '{input.value}' into input field with selector: {selector}"

    except Exception as e:
        return f"Error filling input field: {str(e)}"


@mcp.tool()
async def exploreByRole(input: ExploreByRoleInput) -> str:
    """Explore page sections by ARIA landmark roles. This provides a semantic overview of page structure and helps identify where different types of content are located. Use this to understand page layout before diving into specific sections."""
    ctx = mcp.get_context()
    page = get_active_page(ctx)

    try:
        # Find all elements with the specified role
        elements = await page.locator(f"[role='{input.role}']").all()
        
        # Also check for implicit roles (semantic HTML elements)
        implicit_selectors = {
            'banner': 'header',
            'main': 'main', 
            'navigation': 'nav',
            'contentinfo': 'footer',
            'form': 'form',
            'article': 'article',
            'section': 'section',
            'complementary': 'aside',
            'search': '[role="search"]'  # search is typically explicit
        }
        
        if input.role in implicit_selectors and input.role != 'search':
            implicit_elements = await page.locator(implicit_selectors[input.role]).all()
            elements.extend(implicit_elements)

        if not elements:
            return json.dumps({
                "role": input.role,
                "found": False,
                "message": f"No elements found with role '{input.role}'"
            }, indent=2)

        # Get accessibility info for each element
        role_sections = []
        for i, element in enumerate(elements):
            try:
                # Get the accessibility snapshot for this specific element
                element_handle = await element.element_handle()
                if element_handle:
                    scoped_tree = await page.accessibility.snapshot(root=element_handle)
                    if scoped_tree:
                        # Simplify the output - just key info
                        section_info = {
                            "index": i,
                            "name": scoped_tree.get('name'),
                            "role": scoped_tree.get('role'),
                            "children_count": len(scoped_tree.get('children', [])),
                            "has_interactive_elements": _has_interactive_children(scoped_tree)
                        }
                        role_sections.append(section_info)
            except Exception as e:
                # Skip elements that can't be accessed
                continue

        return json.dumps({
            "role": input.role,
            "found": True,
            "count": len(role_sections),
            "sections": role_sections
        }, indent=2)

    except Exception as e:
        return f"Error exploring by role '{input.role}': {str(e)}"


def _has_interactive_children(node):
    """Check if a node has any interactive children"""
    if not node:
        return False
    
    interactive_roles = {
        'button', 'link', 'textbox', 'combobox', 'listbox', 'option', 
        'checkbox', 'radio', 'tab', 'menuitem', 'treeitem', 'gridcell',
        'columnheader', 'rowheader', 'searchbox', 'slider', 'spinbutton'
    }
    
    role = node.get('role', '').lower()
    if role in interactive_roles:
        return True
    
    for child in node.get('children', []):
        if _has_interactive_children(child):
            return True
    
    return False


if __name__ == "__main__":
    mcp.run()
