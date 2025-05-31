#!/usr/bin/env python3
import json
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
    video_path: str


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage browser lifecycle"""
    # Initialize browser on startup
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()

    try:
        yield AppContext(browser=browser, context=context, page=page, video_path="")
    finally:
        # Cleanup on shutdown
        await context.close()
        await browser.close()


mcp = FastMCP("better-playwright", lifespan=app_lifespan)


class ElementActionInput(BaseModel):
    type: Literal["className", "id", "text"] = Field(
        ..., description="How to select the element"
    )
    selector: str = Field(
        ..., description="The selector value (class name, id, or text content)"
    )
    action: Literal["click", "getText", "extractLinks", "getRawElement", "type"] = Field(
        ..., description="Action to perform"
    )
    text: str | None = Field(
        None, description="Text to type (required when action is 'type')"
    )


class NavigateInput(BaseModel):
    url: str = Field(..., description="URL to navigate to")


class SnapshotInput(BaseModel):
    type: Literal["accessibility", "image"] = Field(
        ..., description="Type of snapshot to capture"
    )


class FillInputInput(BaseModel):
    type: Literal["className", "id", "text", "placeholder", "label"] = Field(
        ..., description="How to select the input element"
    )
    selector: str = Field(
        ...,
        description="The selector value (class name, id, text content, placeholder, or label)",
    )
    value: str = Field(..., description="The text to fill into the input field")


def get_active_page(ctx) -> Page:
    """Get the currently active page"""
    app_context = ctx.request_context.lifespan_context
    return app_context.page


@mcp.tool()
async def getElement(input: ElementActionInput) -> str:
    """Find an element and perform an action on it"""
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
    """Navigate to a URL"""
    ctx = mcp.get_context()
    page = get_active_page(ctx)

    try:
        await page.goto(input.url)
        return f"Successfully navigated to {input.url}"
    except Exception as e:
        return f"Error navigating to {input.url}: {str(e)}"


@mcp.tool()
async def getSnapshot(input: SnapshotInput) -> str:
    """Capture an accessibility tree or image snapshot of the current page. Use this before you click or get an element so that youcan see the state of the page. Use the accessibility tree to get the state of the page and selectors you can use while navigating and the image to see the state of the page."""
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
                    "format": "png",
                    "data": screenshot_base64,
                    "encoding": "base64",
                },
                indent=2,
            )

    except Exception as e:
        return f"Error capturing {input.type} snapshot: {str(e)}"


@mcp.tool()
async def fillInput(input: FillInputInput) -> str:
    """Fill an input field with text using Playwright's fill() method. This is the recommended way to fill form inputs."""
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
