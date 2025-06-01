# Better Playwright MCP Server

A comprehensive Model Context Protocol (MCP) server that provides browser automation capabilities using Playwright. This server enables AI assistants to interact with web pages through actions like navigation, element interaction, form filling, and page analysis.

## Features

- **Browser lifecycle management** with Chrome DevTools Protocol (CDP) connection support
- **Element interaction** - click, type, extract text/links from web elements
- **Navigation and history management** - go to URLs, back/forward, refresh
- **Form input handling** with multiple selector types (ID, class, text, placeholder, label)
- **Page snapshots** - accessibility tree analysis and visual screenshots
- **Automatic browser cleanup** and comprehensive error handling

## Installation

Install the package using uv:

```bash
uv tool install mudae
```

## Usage

Run the MCP server:

```bash
uvx mudae
```

The server will start and be available for MCP client connections.

### Browser Connection

By default, the server attempts to connect to an existing browser instance via Chrome DevTools Protocol (CDP). You can control this behavior with the `LOCAL_CDP_URL` environment variable:

```bash
# Connect to browser running on custom port
LOCAL_CDP_URL=http://localhost:9223 uvx mudae

# Connect to default CDP endpoint (localhost:9222)
uvx mudae
```

If no existing browser is found at the CDP URL, the server will automatically launch a new browser instance at localhost:9222 by default

### Starting a Browser with CDP

To start Chrome/Chromium with CDP enabled for persistent browser sessions:

```bash
# Chrome
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

# Chromium
chromium --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```

## Available Tools

The MCP server provides these tools for browser automation:

### Navigation

- **navigate** - Go to URLs, navigate browser history (back/forward), refresh pages

### Element Interaction

- **getElement** - Find and interact with page elements (click, get text, extract links, type text)
- **fillInput** - Fill form input fields with proper event handling

### Page Analysis

- **getSnapshot** - Capture page state as accessibility trees or visual screenshots
- **exploreByRole** - Explore page sections by ARIA landmark roles

## When to Use

This MCP server is ideal when you need an AI assistant to:

- **Automate web workflows** - Fill forms, click buttons, navigate between pages
- **Extract web content** - Get text, links, and structured data from websites
- **Test web applications** - Verify page content and functionality
- **Analyze page accessibility** - Understand page structure and interactive elements
- **Take visual snapshots** - Capture screenshots for documentation or verification

The server is particularly useful for tasks requiring persistent browser sessions, complex web interactions, or detailed page analysis that goes beyond simple HTTP requests.

## Environment Variables

- `LOCAL_CDP_URL` - Chrome DevTools Protocol endpoint (default: `http://localhost:9222`)

## Requirements

- Python 3.8+
- Playwright (automatically installed with dependencies)

The server handles Playwright browser installation automatically on first run.
