function showLoading() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('syncSubmitBtn').disabled = true;
    document.getElementById('asyncSubmitBtn').disabled = true;
}

function toggleDetail(button) {
    const detail = button.parentElement.nextElementSibling;
    detail.style.display = detail.style.display === 'none' ? 'block' : 'none';
    button.textContent = detail.style.display === 'none' ? '显示详情' : '隐藏详情';
}

// 校准错误位置索引并高亮
function highlightErrors() {
    const essayContainer = document.getElementById('essayContainer');
    const essayContentElement = essayContainer.querySelector('.essay-content');
    
    if (!essayContentElement) {
        console.error('找不到essay-content元素');
        return;
    }
    
    let essayText = essayContentElement.textContent || '';
    let htmlContent = '';
    let lastIndex = 0;
    
    // 获取所有错误并按开始位置排序
    const errors = [];
    const errorElements = document.querySelectorAll('.error-position');
    
    errorElements.forEach(element => {
        const positionText = element.textContent;
        // 匹配位置信息
        const matches = positionText.match(/位置:\s*(\d+)\s*-\s*(\d+)/);
        if (matches && matches.length === 3) {
            const start = parseInt(matches[1]);
            const end = parseInt(matches[2]);
            const errorItem = element.closest('.error-item');
            const errorType = errorItem.closest('.error-category').querySelector('.error-type').textContent;
            
            errors.push({
                start,
                end,
                type: errorType,
                element: errorItem
            });
        }
    });
    
    // 排序错误，从前到后处理
    errors.sort((a, b) => a.start - b.start);
    
    // 处理每个错误
    errors.forEach(error => {
        // 添加错误前的正常文本
        if (error.start > lastIndex) {
            const normalText = essayText.substring(lastIndex, error.start);
            htmlContent += preserveWhitespace(normalText);
        }
        
        // 添加带高亮的错误文本
        let errorText = '';
        if (error.start < essayText.length && error.end <= essayText.length) {
            errorText = essayText.substring(error.start, error.end);
        } else {
            console.warn(`错误索引超出范围: ${error.start}-${error.end}, 文本长度: ${essayText.length}`);
            // 尝试从错误元素中获取错误文本
            const errorTextElement = error.element.querySelector('.error-text-content');
            if (errorTextElement) {
                errorText = errorTextElement.textContent.replace(/^"(.*)"$/, '$1'); // 移除引号
            }
        }
        
        htmlContent += `<span class="highlight-error" 
            data-type="${error.type}" 
            data-error-id="${error.element.id || ''}"
            data-start="${error.start}"
            data-end="${error.end}">${preserveWhitespace(errorText)}</span>`;
        
        lastIndex = error.end;
    });
    
    // 添加最后一个错误后的正常文本
    if (lastIndex < essayText.length) {
        const normalText = essayText.substring(lastIndex);
        htmlContent += preserveWhitespace(normalText);
    }
    
    essayContentElement.innerHTML = htmlContent;
    
    // 添加错误点击交互
    setupErrorInteractions();
}

// 保留空白字符和换行
function preserveWhitespace(text) {
    return text
        .replace(/\n/g, '<br>')
        .replace(/ /g, '&nbsp;')
        .replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');
}

// 转义HTML特殊字符
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 设置错误交互功能
function setupErrorInteractions() {
    // 添加错误悬停提示
    document.querySelectorAll('.highlight-error').forEach(errorSpan => {
        const tooltipId = `tooltip-${Math.random().toString(36).substr(2, 9)}`;
        
        // 悬停显示错误类型
        errorSpan.addEventListener('mouseover', function() {
            // 检查是否已经存在tooltip
            if (this.querySelector('.error-tooltip')) return;
            
            const errorType = this.getAttribute('data-type');
            const tooltip = document.createElement('div');
            tooltip.className = 'error-tooltip';
            tooltip.id = tooltipId;
            tooltip.textContent = `错误类型: ${errorType}`;
            this.appendChild(tooltip);
            
            // 高亮对应的错误项
            const errorId = this.getAttribute('data-error-id');
            if (errorId) {
                document.getElementById(errorId).classList.add('active-error');
            }
        });

        // 鼠标移出时移除提示
        errorSpan.addEventListener('mouseout', function() {
            const tooltip = document.getElementById(tooltipId);
            if (tooltip) {
                tooltip.remove();
            }
            
            // 移除高亮
            const errorId = this.getAttribute('data-error-id');
            if (errorId) {
                document.getElementById(errorId).classList.remove('active-error');
            }
        });
        
        // 点击跳转到对应的错误详情
        errorSpan.addEventListener('click', function() {
            const errorId = this.getAttribute('data-error-id');
            if (errorId) {
                const errorItem = document.getElementById(errorId);
                errorItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // 展开错误详情
                const detailButton = errorItem.querySelector('.toggle-detail');
                const detail = errorItem.querySelector('.error-detail');
                if (detail.style.display === 'none') {
                    detail.style.display = 'block';
                    detailButton.textContent = '隐藏详情';
                }
                
                // 添加闪烁效果
                this.classList.add('highlight-pulse');
                setTimeout(() => {
                    this.classList.remove('highlight-pulse');
                }, 2000);
            }
        });
    });
}

// 动画效果
function initAnimations() {
    // 添加进入动画
    const sections = document.querySelectorAll('.section');
    if (sections.length) {
        sections.forEach((section, index) => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
            }, 100 * index);
        });
    }
    
    // 添加评分卡片动画
    const scoreCard = document.querySelector('.score-card');
    if (scoreCard) {
        scoreCard.style.opacity = '0';
        scoreCard.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            scoreCard.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            scoreCard.style.opacity = '1';
            scoreCard.style.transform = 'scale(1)';
        }, 300);
    }
}

// 显示加载状态
function showLoading() {
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submitBtn');
    
    if (loading) loading.style.display = 'block';
    if (submitBtn) submitBtn.disabled = true;
}

// 隐藏加载状态
function hideLoading() {
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submitBtn');
    
    if (loading) loading.style.display = 'none';
    if (submitBtn) submitBtn.disabled = false;
}

// 初始化页面
document.addEventListener('DOMContentLoaded', function() {
    // 自动折叠所有错误详情
    document.querySelectorAll('.error-detail').forEach((detail, index) => {
        detail.style.display = 'none';
        
        // 为每个错误项添加ID
        const errorItem = detail.closest('.error-item');
        if (errorItem) {
            errorItem.id = `error-item-${index}`;
        }
    });
    
    // 初始化高亮
    highlightErrors();
    
    // 监听DOM变化，自动重新初始化错误高亮
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.type === 'childList' && 
                (mutation.target.classList.contains('essay-content') || 
                mutation.target.classList.contains('error-container'))) {
                highlightErrors();
                break;
            }
        }
    });
    
    // 观察essay-content和error-container的变化
    const essayContent = document.querySelector('.essay-content');
    const errorContainer = document.querySelector('.error-container');
    if (essayContent) {
        observer.observe(essayContent, { childList: true, subtree: true });
    }
    if (errorContainer) {
        observer.observe(errorContainer, { childList: true, subtree: true });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    // 初始化表单提交处理
    initFormSubmission();
    
    // 初始化示例文章加载
    initSampleEssays();
    
    // 初始化错误高亮功能
    initErrorHighlighting();
    
    // 初始化详情切换功能
    initToggleDetails();
    
    // 初始化动画效果
    initAnimations();
});

// 表单提交处理
function initFormSubmission() {
    const form = document.getElementById('essayForm');
    if (!form) return;
    
    form.addEventListener('submit', function(e) {
        const processType = document.querySelector('input[name="process_type"]:checked').value;
        
        // 如果不是同步处理，阻止表单提交，改用AJAX
        if (processType !== 'sync') {
            e.preventDefault();
            
            const essay = document.getElementById('essay').value;
            const engineType = document.querySelector('input[name="engine_type"]:checked').value;
            
            if (!essay.trim()) {
                showError('请输入作文内容');
                return;
            }
            
            // 显示加载状态
            showLoading();
            
            // 根据处理类型发送不同的请求
            if (processType === 'async') {
                submitAsync(essay, engineType);
            } else if (processType === 'stream') {
                submitStream(essay, engineType);
            }
        } else {
            // 同步提交也显示加载状态
            showLoading();
        }
    });
}

// 异步提交处理
function submitAsync(essay, engineType) {
    fetch('/api/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            essay: essay,
            engine_type: engineType
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            hideLoading();
            showError(data.error);
        } else {
            // 重定向到等待页面
            window.location.href = '/async-result/' + data.task_id;
        }
    })
    .catch(error => {
        hideLoading();
        showError('提交失败: ' + error.message);
    });
}

// 流式提交处理
function submitStream(essay, engineType) {
    fetch('/api/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            essay: essay,
            engine_type: engineType
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            hideLoading();
            showError(data.error);
        } else {
            // 重定向到流式结果页面
            window.location.href = '/stream-result/' + data.task_id;
        }
    })
    .catch(error => {
        hideLoading();
        showError('提交失败: ' + error.message);
    });
}

// 示例文章加载
function initSampleEssays() {
    const sampleBtns = document.querySelectorAll('.sample-btn');
    if (!sampleBtns.length) return;
    
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const sampleId = this.dataset.sample;
            const essayTextarea = document.getElementById('essay');
            
            // 添加加载效果
            this.classList.add('loading');
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 加载中...';
            
            // 从API获取示例作文内容
            fetch(`/api/sample/${sampleId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.content) {
                        essayTextarea.value = data.content;
                        
                        // 平滑滚动到表单
                        document.querySelector('.form-group').scrollIntoView({ 
                            behavior: 'smooth' 
                        });
                        
                        // 聚焦文本区域
                        essayTextarea.focus();
                    } else {
                        showError('无法加载示例作文');
                    }
                    
                    // 恢复按钮状态
                    resetSampleButtons();
                })
                .catch(error => {
                    showError('加载示例作文失败: ' + error.message);
                    resetSampleButtons();
                });
        });
    });
}

// 重置示例按钮状态
function resetSampleButtons() {
    const sampleBtns = document.querySelectorAll('.sample-btn');
    sampleBtns.forEach((btn, index) => {
        btn.classList.remove('loading');
        btn.innerHTML = `示例${index + 1}: ${index === 0 ? '中学生作文' : '大学生作文'}`;
    });
}

// 错误高亮功能
function initErrorHighlighting() {
    const essayContainer = document.getElementById('essayContainer');
    if (!essayContainer) return;
    
    // 获取所有错误项
    const errorItems = document.querySelectorAll('.error-item');
    
    errorItems.forEach(item => {
        const errorHeader = item.querySelector('.error-header');
        
        errorHeader.addEventListener('click', function(e) {
            // 如果点击的是切换详情按钮，不执行高亮
            if (e.target.classList.contains('toggle-detail')) return;
            
            // 获取错误位置
            const posValue = item.querySelector('.error-pos-value').textContent;
            const positions = posValue.split(' - ').map(Number);
            
            if (positions.length === 2) {
                highlightErrorInEssay(positions[0], positions[1]);
                
                // 添加活跃状态到当前错误项
                document.querySelectorAll('.error-item').forEach(el => {
                    el.classList.remove('active-error');
                });
                item.classList.add('active-error');
            }
        });
    });
}

// 在作文中高亮错误
function highlightErrorInEssay(start, end) {
    const essayContainer = document.getElementById('essayContainer');
    const essayContentElement = essayContainer.querySelector('.essay-content');
    if (!essayContentElement) return;
    
    // 移除之前的pulse效果
    document.querySelectorAll('.highlight-pulse').forEach(el => {
        el.classList.remove('highlight-pulse');
    });
    
    // 找到对应的错误span
    const errorSpans = document.querySelectorAll('.highlight-error');
    let targetSpan = null;
    
    for (const span of errorSpans) {
        const spanStart = parseInt(span.getAttribute('data-start'));
        const spanEnd = parseInt(span.getAttribute('data-end'));
        
        if (spanStart === start && spanEnd === end) {
            targetSpan = span;
            break;
        }
    }
    
    if (targetSpan) {
        // 添加pulse效果到目标span
        targetSpan.classList.add('highlight-pulse');
    
    // 滚动到高亮位置
        targetSpan.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }
}

// 详情切换功能
function initToggleDetails() {
    // 这个函数已在HTML中直接实现，这里作为备用
    window.toggleDetail = function(button) {
        const detail = button.parentElement.nextElementSibling;
        if (detail.style.display === 'block') {
            detail.style.display = 'none';
            button.textContent = '显示详情';
        } else {
            detail.style.display = 'block';
            button.textContent = '隐藏详情';
        }
    };
}

// 动画效果
function initAnimations() {
    // 添加进入动画
    const sections = document.querySelectorAll('.section');
    if (sections.length) {
        sections.forEach((section, index) => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
            }, 100 * index);
        });
    }
    
    // 添加评分卡片动画
    const scoreCard = document.querySelector('.score-card');
    if (scoreCard) {
        scoreCard.style.opacity = '0';
        scoreCard.style.transform = 'scale(0.95)';
        
        setTimeout(() => {
            scoreCard.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            scoreCard.style.opacity = '1';
            scoreCard.style.transform = 'scale(1)';
        }, 300);
    }
}

// 显示加载状态
function showLoading() {
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submitBtn');
    
    if (loading) loading.style.display = 'block';
    if (submitBtn) submitBtn.disabled = true;
}

// 隐藏加载状态
function hideLoading() {
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submitBtn');
    
    if (loading) loading.style.display = 'none';
    if (submitBtn) submitBtn.disabled = false;
}

// 显示错误信息
function showError(message) {
    // 检查是否已有错误提示
    let errorAlert = document.querySelector('.alert.error');
    
    if (!errorAlert) {
        // 创建新的错误提示
        errorAlert = document.createElement('div');
        errorAlert.className = 'alert error';
        
        const form = document.getElementById('essayForm');
        if (form) {
            form.parentNode.insertBefore(errorAlert, form);
        }
    }
    
    // 设置错误消息
    errorAlert.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    
    // 滚动到错误消息
    errorAlert.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // 3秒后自动隐藏
    setTimeout(() => {
        errorAlert.style.opacity = '0';
        setTimeout(() => {
            if (errorAlert.parentNode) {
                errorAlert.parentNode.removeChild(errorAlert);
            }
        }, 300);
    }, 3000);
    
    // 添加过渡效果
    errorAlert.style.transition = 'opacity 0.3s ease';
}
