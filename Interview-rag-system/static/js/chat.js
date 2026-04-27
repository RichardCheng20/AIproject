// 聊天界面JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const sourcesList = document.getElementById('sources-list');
    const toggleSources = document.getElementById('toggle-sources');
    const clearChatBtn = document.getElementById('clear-chat');
    const refreshKbBtn = document.getElementById('refresh-kb');
    const newChatBtn = document.getElementById('new-chat');
    const exampleButtons = document.querySelectorAll('.example-btn');
    const streamToggle = document.getElementById('stream-toggle');
    const typingIndicator = document.getElementById('typing-indicator');
    const currentTimeEl = document.getElementById('current-time');

    // 当前会话状态
    let sessionId = generateSessionId();
    let isStreaming = streamToggle.checked;
    let isProcessing = false;

    // 初始化
    init();

    function init() {
        // 设置当前时间
        updateCurrentTime();
        setInterval(updateCurrentTime, 60000);

        // 加载系统信息
        loadSystemInfo();

        // 设置事件监听器
        setupEventListeners();

        // 调整输入框高度
        adjustTextareaHeight(messageInput);
    }

    function setupEventListeners() {
        // 发送消息
        sendButton.addEventListener('click', sendMessage);

        // 输入框回车发送
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // 输入框自动调整高度
        messageInput.addEventListener('input', function() {
            adjustTextareaHeight(this);
        });

        // 示例问题按钮
        exampleButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                const question = this.getAttribute('data-question');
                messageInput.value = question;
                adjustTextareaHeight(messageInput);
                sendMessage();
            });
        });

        // 清空对话
        clearChatBtn.addEventListener('click', clearChat);

        // 刷新知识库
        refreshKbBtn.addEventListener('click', refreshKnowledgeBase);

        // 新对话
        newChatBtn.addEventListener('click', startNewChat);

        // 切换来源面板
        toggleSources.addEventListener('click', function() {
            const sourcesPanel = document.querySelector('.sources-panel');
            sourcesPanel.classList.toggle('active');
            const icon = this.querySelector('i');
            icon.classList.toggle('fa-chevron-left');
            icon.classList.toggle('fa-chevron-right');
        });

        // 流式响应切换
        streamToggle.addEventListener('change', function() {
            isStreaming = this.checked;
        });
    }

    function sendMessage() {
        const message = messageInput.value.trim();

        if (!message || isProcessing) {
            return;
        }

        // 添加用户消息
        addMessage(message, 'user');

        // 清空输入框
        messageInput.value = '';
        adjustTextareaHeight(messageInput);

        // 显示正在输入指示器
        showTypingIndicator();

        // 发送请求
        if (isStreaming) {
            sendStreamingRequest(message);
        } else {
            sendNormalRequest(message);
        }
    }

    function sendNormalRequest(question) {
        isProcessing = true;

        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                session_id: sessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();

            if (data.status === 'success') {
                addMessage(data.answer, 'assistant');
                updateSources(data.relevant_sources);
            } else {
                addMessage(`抱歉，出现错误：${data.error || '未知错误'}`, 'assistant');
            }

            isProcessing = false;
        })
        .catch(error => {
            hideTypingIndicator();
            addMessage(`请求失败：${error.message}`, 'assistant');
            isProcessing = false;
            console.error('Error:', error);
        });
    }

    function sendStreamingRequest(question) {
        isProcessing = true;

        fetch('/api/query/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                session_id: sessionId,
                stream: true
            })
        })
        .then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            // 创建消息容器
            const messageId = addMessage('', 'assistant', true);
            const messageElement = document.getElementById(`message-${messageId}`);
            const textElement = messageElement.querySelector('.message-text');

            hideTypingIndicator();

            function read() {
                reader.read().then(({done, value}) => {
                    if (done) {
                        isProcessing = false;

                        // 请求来源信息
                        fetchSources(question);
                        return;
                    }

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    lines.forEach(line => {
                        if (line.startsWith('data: ')) {
                            const text = line.substring(6);
                            textElement.innerHTML += escapeHtml(text);
                        }
                    });

                    // 滚动到底部
                    scrollToBottom();

                    // 继续读取
                    read();
                })
                .catch(error => {
                    console.error('Stream reading error:', error);
                    isProcessing = false;
                });
            }

            read();
        })
        .catch(error => {
            hideTypingIndicator();
            addMessage(`流式请求失败：${error.message}`, 'assistant');
            isProcessing = false;
            console.error('Error:', error);
        });
    }

    function fetchSources(question) {
        // 对于流式响应，单独获取来源信息
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                session_id: sessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateSources(data.relevant_sources);
            }
        })
        .catch(error => {
            console.error('Failed to fetch sources:', error);
        });
    }

    function addMessage(text, sender, isStreaming = false) {
        const messageId = Date.now();
        const timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

        const messageHtml = `
            <div class="message ${sender}-message" id="message-${messageId}">
                <div class="message-avatar">
                    <i class="fas ${sender === 'user' ? 'fa-user' : 'fa-robot'}"></i>
                </div>
                <div class="message-content">
                    <div class="message-header">
                        <strong>${sender === 'user' ? '您' : '面试助手'}</strong>
                        <span class="message-time">${timestamp}</span>
                    </div>
                    <div class="message-text">
                        ${isStreaming ? '' : escapeHtml(text).replace(/\n/g, '<br>')}
                    </div>
                </div>
            </div>
        `;

        chatMessages.insertAdjacentHTML('beforeend', messageHtml);
        scrollToBottom();

        return messageId;
    }

    function updateSources(sources) {
        if (!sources || sources.length === 0) {
            sourcesList.innerHTML = '<p class="sources-empty">暂无来源信息</p>';
            return;
        }

        let sourcesHtml = '';

        sources.forEach((source, index) => {
            const sourceName = source.source || '未知来源';
            const sourceType = source.source_type || 'other';
            const content = source.content || '';
            const chunkId = source.chunk_id || index + 1;
            const sourceTypeLabel = sourceType === 'resume'
                ? '候选人简历'
                : (sourceType === 'additional' ? '补充资料' : '其他资料');

            sourcesHtml += `
                <div class="source-item">
                    <div class="source-header">
                        <span class="source-title">${escapeHtml(sourceName)}</span>
                        <span class="source-id">#${chunkId}</span>
                    </div>
                    <div class="source-content">
                        ${escapeHtml(content)}
                    </div>
                    <div class="source-footer">
                        <span>相关度: ${source.relevance_score ? source.relevance_score.toFixed(2) : 'N/A'}</span>
                        <span>来源类型: ${sourceTypeLabel}</span>
                    </div>
                </div>
            `;
        });

        sourcesList.innerHTML = sourcesHtml;
    }

    function clearChat() {
        if (!confirm('确定要清空当前对话吗？')) {
            return;
        }

        // 保留欢迎消息
        const welcomeMessage = chatMessages.querySelector('.system-message');
        chatMessages.innerHTML = '';

        if (welcomeMessage) {
            chatMessages.appendChild(welcomeMessage);
        }

        // 清空来源
        sourcesList.innerHTML = '<p class="sources-empty">暂无来源信息</p>';

        // 生成新会话ID
        sessionId = generateSessionId();
    }

    function startNewChat() {
        clearChat();
    }

    function refreshKnowledgeBase() {
        if (isProcessing) {
            alert('系统正在处理请求，请稍后再试。');
            return;
        }

        if (!confirm('刷新知识库需要一些时间，确定要继续吗？')) {
            return;
        }

        fetch('/api/knowledge-base/refresh', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            alert('知识库刷新任务已启动，请稍后查看系统状态。');
        })
        .catch(error => {
            alert('刷新请求失败：' + error.message);
            console.error('Error:', error);
        });
    }

    function loadSystemInfo() {
        fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('model-info').textContent = data.model;

            // 加载知识库信息
            fetch('/api/knowledge-base/info')
            .then(response => response.json())
            .then(kbData => {
                const count = kbData.document_count || 0;
                document.getElementById('kb-info').textContent = `${count} 个文档块`;
            })
            .catch(error => {
                console.error('Failed to load KB info:', error);
            });
        })
        .catch(error => {
            console.error('Failed to load system info:', error);
        });
    }

    function showTypingIndicator() {
        typingIndicator.style.display = 'flex';
        scrollToBottom();
    }

    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function adjustTextareaHeight(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    function updateCurrentTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        const dateString = now.toLocaleDateString('zh-CN', {year: 'numeric', month: 'long', day: 'numeric'});

        currentTimeEl.textContent = `${dateString} ${timeString}`;
    }

    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 初始滚动到底部
    setTimeout(scrollToBottom, 100);
});