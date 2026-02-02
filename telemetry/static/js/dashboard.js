// Telemetry Dashboard JavaScript
let ws = null;
let activityLog = [];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('Telemetry Dashboard initializing...');

    // Load initial data
    loadMetrics();
    loadConversations();
    loadRAGCollections();

    // Connect WebSocket for live updates
    connectWebSocket();

    // Refresh data periodically
    setInterval(loadMetrics, 5000);
    setInterval(loadConversations, 10000);
});

// Connect to WebSocket for real-time updates
function connectWebSocket() {
    const wsUrl = `ws://${window.location.hostname}:8001/ws/metrics`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        updateStatus('Live', true);
        addActivity('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateMetrics(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Error', false);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateStatus('Disconnected', false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
    };
}

// Load metrics from API
async function loadMetrics() {
    try {
        const response = await fetch('/api/telemetry/metrics');
        const data = await response.json();
        updateMetrics(data);
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// Update metrics display
function updateMetrics(data) {
    if (data.database) {
        document.getElementById('db-conversations').textContent = data.database.conversations || 0;
        document.getElementById('db-messages').textContent = data.database.messages || 0;
        document.getElementById('db-invocations').textContent = data.database.agent_invocations || 0;
        document.getElementById('db-orders').textContent = data.database.orders || 0;
    }

    if (data.rag) {
        document.getElementById('rag-total-docs').textContent = data.rag.total_documents || 0;
        document.getElementById('rag-collections').textContent = data.rag.collections || 0;

        // Update collections list
        if (data.rag.collections_info) {
            updateCollectionsList(data.rag.collections_info);
        }
    }

    if (data.agents) {
        document.getElementById('agent-count').textContent = data.agents.total_agents || 0;

        // Update agents list
        if (data.agents.registered_agents) {
            updateAgentsList(data.agents.registered_agents);
        }
    }
}

// Update collections list
function updateCollectionsList(collections) {
    const container = document.getElementById('collections-list');
    container.innerHTML = '';

    for (const [name, count] of Object.entries(collections)) {
        const item = document.createElement('div');
        item.className = 'collection-item';
        item.innerHTML = `
            <span class="collection-name">${name.replace(/_/g, ' ')}</span>
            <span class="collection-count">${count} docs</span>
        `;
        container.appendChild(item);
    }
}

// Update agents list
function updateAgentsList(agents) {
    const container = document.getElementById('agents-list');
    container.innerHTML = '';

    agents.forEach(agent => {
        const item = document.createElement('div');
        item.className = 'agent-item';
        item.innerHTML = `
            <span class="agent-name">${agent.replace(/_/g, ' ')}</span>
            <span class="agent-status">●</span>
        `;
        container.appendChild(item);
    });
}

// Load conversations
async function loadConversations() {
    try {
        const response = await fetch('/api/telemetry/conversations');
        const data = await response.json();
        updateConversationsTable(data.recent || []);
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

// Update conversations table
function updateConversationsTable(conversations) {
    const tbody = document.getElementById('conversations-tbody');

    if (conversations.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="loading">No conversations yet</td></tr>';
        return;
    }

    tbody.innerHTML = conversations.map(conv => `
        <tr>
            <td><code>${conv.id ? conv.id.substring(0, 20) : 'N/A'}...</code></td>
            <td>${conv.started_at ? new Date(conv.started_at).toLocaleString() : 'N/A'}</td>
            <td><span class="status-badge">${conv.status || 'active'}</span></td>
            <td>${conv.message_count || 0}</td>
            <td><button class="btn" onclick="viewConversation('${conv.id}')">View</button></td>
        </tr>
    `).join('');
}

// Load RAG collections
async function loadRAGCollections() {
    try {
        const response = await fetch('/api/telemetry/rag/collections');
        const data = await response.json();
        updateRAGTable(data);
    } catch (error) {
        console.error('Error loading RAG collections:', error);
    }
}

// Update RAG table
function updateRAGTable(collections) {
    const tbody = document.getElementById('rag-tbody');

    const entries = Object.values(collections);
    if (entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" class="loading">No collections</td></tr>';
        return;
    }

    tbody.innerHTML = entries.map(coll => `
        <tr>
            <td><strong>${coll.name.replace(/_/g, ' ')}</strong></td>
            <td>${coll.document_count}</td>
            <td>${new Date(coll.last_updated).toLocaleString()}</td>
        </tr>
    `).join('');
}

// View conversation details
async function viewConversation(id) {
    addActivity(`Viewing conversation: ${id}`);

    try {
        const response = await fetch(`/api/conversations/${id}`);
        const data = await response.json();

        if (data.messages && data.messages.length > 0) {
            // Format messages for display
            let messageText = `Conversation: ${id}\n\n`;
            data.messages.forEach((msg, idx) => {
                messageText += `${idx + 1}. [${msg.role.toUpperCase()}]: ${msg.content.substring(0, 200)}...\n\n`;
            });

            // Show in alert (could be improved with a modal)
            alert(messageText);
            addActivity(`Loaded ${data.messages.length} messages from conversation`);
        } else {
            alert('No messages found in this conversation');
        }
    } catch (error) {
        console.error('Error loading conversation:', error);
        alert('Error loading conversation details');
        addActivity(`Error loading conversation: ${error.message}`);
    }
}

// Update status indicator
function updateStatus(text, isLive) {
    document.getElementById('status-text').textContent = text;
    const dot = document.querySelector('.status-dot');
    const indicator = document.querySelector('.status-indicator');

    if (isLive) {
        dot.style.background = '#4caf50';
        indicator.style.borderColor = 'rgba(76, 175, 80, 0.3)';
        indicator.style.background = 'rgba(76, 175, 80, 0.1)';
    } else {
        dot.style.background = '#f44336';
        indicator.style.borderColor = 'rgba(244, 67, 54, 0.3)';
        indicator.style.background = 'rgba(244, 67, 54, 0.1)';
    }
}

// Add activity log entry
function addActivity(message) {
    const log = document.getElementById('activity-log');
    const time = new Date().toLocaleTimeString();

    const item = document.createElement('div');
    item.className = 'activity-item';
    item.innerHTML = `
        <span class="activity-time">${time}</span>
        <span class="activity-message">${message}</span>
    `;

    log.insertBefore(item, log.firstChild);

    // Keep only last 20 items
    while (log.children.length > 20) {
        log.removeChild(log.lastChild);
    }
}

// Format numbers
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}
