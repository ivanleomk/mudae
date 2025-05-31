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
    type: Literal["accessibility", "image"] = Field(
        ...,
        description="Type of snapshot: 'accessibility' to get the accessibility tree structure for finding elements, 'image' to capture a visual screenshot of the page",
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


@mcp.tool()
async def getSnapshot(input: SnapshotInput) -> str:
    """Capture the current state of the page either as an accessibility tree or visual screenshot. The accessibility tree shows all interactive elements and their roles/names - perfect for finding selectors before interacting with elements. Screenshots provide visual confirmation of page state. Always use this before attempting to interact with page elements."""
    ctx = mcp.get_context()
    page = get_active_page(ctx)

    try:
        if input.type == "accessibility":
            # Get accessibility tree snapshot
            accessibility_tree = await page.accessibility.snapshot()
            return json.dumps(
                {"type": "accessibility", "snapshot": accessibility_tree}, indent=2
            )

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


if __name__ == "__main__":
    mcp.run()
