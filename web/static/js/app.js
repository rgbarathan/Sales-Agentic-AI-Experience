// Agentic Sales System - Frontend JavaScript

let ws = null;
let conversationId = null;
let messageCount = 0;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeWebSocket();
    loadAgentStatus();
    autoResizeTextarea();
});

// Initialize WebSocket connection
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        updateStatus('connected', 'Connected');
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleResponse(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('error', 'Connection Error');
    };

    ws.onclose = () => {
        updateStatus('disconnected', 'Disconnected');
        console.log('WebSocket closed');
        // Attempt to reconnect after 3 seconds
        setTimeout(initializeWebSocket, 3000);
    };
}

// Update connection status
function updateStatus(status, text) {
    const indicator = document.getElementById('status');
    const statusText = document.getElementById('status-text');

    // Elements were removed from UI - just log instead
    if (!indicator || !statusText) {
        console.log('WebSocket status:', status, text);
        return;
    }

    indicator.className = 'status-indicator';

    if (status === 'connected') {
        indicator.style.background = 'var(--success)';
    } else if (status === 'error') {
        indicator.style.background = 'var(--error)';
    } else {
        indicator.style.background = 'var(--warning)';
    }

    statusText.textContent = text;
}

// Send message
function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();

    if (!message || !ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }

    // Add user message to chat
    addMessage('user', message);

    // Send via WebSocket
    ws.send(JSON.stringify({
        message: message,
        conversation_id: conversationId
    }));

    // Clear input
    input.value = '';
    input.style.height = 'auto';

    // Disable send button temporarily
    const sendButton = document.getElementById('send-button');
    sendButton.disabled = true;
}

// Send quick message
function sendQuickMessage(message) {
    const input = document.getElementById('message-input');
    input.value = message;
    sendMessage();
}

// Handle response from server
function handleResponse(data) {
    if (data.error) {
        console.error('Server error:', data.error);
        addMessage('assistant', `Error: ${data.error}`);
        return;
    }

    // Update conversation ID
    if (data.conversation_id) {
        conversationId = data.conversation_id;
        const convIdEl = document.getElementById('conversation-id');
        if (convIdEl) {
            convIdEl.textContent = conversationId.substring(0, 20) + '...';
        }
    }

    // Add assistant message with agent activity
    if (data.message) {
        addMessage('assistant', data.message, data.agent_activity);
    }

    // Re-enable send button
    const sendButton = document.getElementById('send-button');
    sendButton.disabled = false;
}

// Add message to chat
function addMessage(role, content, agentActivity = null) {
    const messagesContainer = document.getElementById('chat-messages');

    // Keep welcome message visible - don't remove it

    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Render markdown to HTML
    const htmlContent = marked.parse(content);
    contentDiv.innerHTML = htmlContent;

    // Add agent activity tags if present
    if (agentActivity && role === 'assistant') {
        const tagsDiv = document.createElement('div');
        tagsDiv.className = 'agent-tags';
        tagsDiv.style.cssText = 'margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px;';

        // Add sub-agents invoked
        if (agentActivity.sub_agents_invoked && agentActivity.sub_agents_invoked.length > 0) {
            agentActivity.sub_agents_invoked.forEach(agent => {
                const tag = document.createElement('span');
                tag.style.cssText = 'background: rgba(102, 126, 234, 0.2); color: #a5b4fc; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(102, 126, 234, 0.3);';
                tag.textContent = `🤖 ${agent}`;
                tagsDiv.appendChild(tag);
            });
        }

        // Add tools used
        if (agentActivity.tools_used && agentActivity.tools_used.length > 0) {
            agentActivity.tools_used.forEach(tool => {
                const tag = document.createElement('span');
                tag.style.cssText = 'background: rgba(34, 197, 94, 0.2); color: #86efac; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(34, 197, 94, 0.3);';
                tag.textContent = `⚡ ${tool}`;
                tagsDiv.appendChild(tag);
            });
        }

        // Add communication methods
        if (agentActivity.communication_methods && agentActivity.communication_methods.length > 0) {
            agentActivity.communication_methods.forEach(method => {
                const tag = document.createElement('span');
                tag.style.cssText = 'background: rgba(251, 146, 60, 0.2); color: #fdba74; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(251, 146, 60, 0.3);';
                tag.textContent = `📡 ${method}`;
                tagsDiv.appendChild(tag);
            });
        }

        if (tagsDiv.children.length > 0) {
            contentDiv.appendChild(tagsDiv);
        }
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Update message count
    messageCount++;
    const messageCountEl = document.getElementById('message-count');
    if (messageCountEl) {
        messageCountEl.textContent = messageCount;
    }
}

// Handle keyboard shortcuts
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Auto-resize textarea
function autoResizeTextarea() {
    const textarea = document.getElementById('message-input');

    textarea.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}

// Load agent status
async function loadAgentStatus() {
    try {
        const response = await fetch('/api/agents/status');
        const data = await response.json();

        const agentList = document.getElementById('agent-status');
        agentList.innerHTML = '';

        if (data.registered_agents && data.registered_agents.length > 0) {
            data.registered_agents.forEach(agent => {
                const agentItem = document.createElement('div');
                agentItem.className = 'agent-item';

                const indicator = document.createElement('div');
                indicator.className = 'agent-indicator';

                const name = document.createElement('span');
                name.textContent = formatAgentName(agent);

                agentItem.appendChild(indicator);
                agentItem.appendChild(name);
                agentList.appendChild(agentItem);
            });
        } else {
            agentList.innerHTML = '<div class="loading">No agents registered</div>';
        }
    } catch (error) {
        console.error('Failed to load agent status:', error);
        document.getElementById('agent-status').innerHTML =
            '<div class="loading">Failed to load</div>';
    }
}

// Format agent name for display
function formatAgentName(agentName) {
    return agentName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Refresh agent status every 30 seconds
setInterval(loadAgentStatus, 30000);
