function render({ model, el }) {
    const container = document.createElement('div');
    container.className = 'astro-widget';

    const header = document.createElement('div');
    header.className = 'astro-header';
    header.textContent = '🚀 AstroLab Widget';
    container.appendChild(header);

    // Ops status indicator
    const opsInfo = document.createElement('div');
    opsInfo.className = 'astro-ops-info';
    opsInfo.innerHTML = '⚡ widget.ops - Direct Blender Access Available';
    container.appendChild(opsInfo);

    // Ops Editor Panel
    const opsEditor = document.createElement('div');
    opsEditor.className = 'astro-ops-editor';
    opsEditor.style.display = 'none'; // Initially hidden

    const opsEditorHeader = document.createElement('div');
    opsEditorHeader.className = 'astro-ops-editor-header';
    opsEditorHeader.innerHTML = '🔧 Blender Ops Editor';
    opsEditor.appendChild(opsEditorHeader);

    const opsEditorContent = document.createElement('div');
    opsEditorContent.className = 'astro-ops-editor-content';
    opsEditor.appendChild(opsEditorContent);

    container.appendChild(opsEditor);

    // Quick scene controls
    const sceneControls = document.createElement('div');
    sceneControls.className = 'astro-scene-controls';

    const quickButtons = [
        { text: 'Galaxy', action: 'galaxy' },
        { text: 'Solar System', action: 'solar_system' },
        { text: 'Reset Scene', action: 'reset' },
        { text: 'Quick Render', action: 'render' },
        { text: 'Ops Editor', action: 'toggle_editor' }
    ];

    quickButtons.forEach(btn => {
        const button = document.createElement('button');
        button.className = 'astro-quick-button';
        button.textContent = btn.text;
        button.onclick = () => {
            if (btn.action === 'toggle_editor') {
                opsEditor.style.display = opsEditor.style.display === 'none' ? 'block' : 'none';
                if (opsEditor.style.display === 'block') {
                    loadOpsEditor();
                }
            } else {
                model.set('trigger_render', model.get('trigger_render') + 1);
                model.set('current_action', btn.action);
                model.save_changes();
            }
        };
        sceneControls.appendChild(button);
    });

    container.appendChild(sceneControls);

    // Main controls
    const controls = document.createElement('div');
    controls.className = 'astro-controls';

    const renderButton = document.createElement('button');
    renderButton.className = 'astro-button';
    renderButton.textContent = '🎬 Render Scene';
    renderButton.onclick = () => {
        model.set('trigger_render', model.get('trigger_render') + 1);
        model.save_changes();
    };
    controls.appendChild(renderButton);

    const clearButton = document.createElement('button');
    clearButton.className = 'astro-button';
    clearButton.textContent = '🧹 Clear';
    clearButton.onclick = () => {
        model.set('image_data', '');
        model.set('image_path', '');
        model.save_changes();
    };
    controls.appendChild(clearButton);

    container.appendChild(controls);

    // Image container
    const imageContainer = document.createElement('div');
    imageContainer.className = 'astro-image-container';

    const updateImage = () => {
        const imageData = model.get('image_data');
        const imagePath = model.get('image_path');

        if (imageData) {
            imageContainer.innerHTML = `<img src="data:image/png;base64,${imageData}" class="astro-image" alt="Rendered Image">`;
        } else if (imagePath) {
            imageContainer.innerHTML = `<img src="${imagePath}" class="astro-image" alt="Rendered Image">`;
        } else {
            imageContainer.innerHTML = '<div class="astro-placeholder">🌌 Ready for astronomical visualization</div>';
        }
    };

    updateImage();
    model.on('change:image_data', updateImage);
    model.on('change:image_path', updateImage);

    container.appendChild(imageContainer);

    // Info panel
    const infoPanel = document.createElement('div');
    infoPanel.className = 'astro-info';

    const updateInfo = () => {
        const renderTime = model.get('render_time');
        const resolution = model.get('resolution');
        const engine = model.get('render_engine');
        const samples = model.get('samples');

        infoPanel.innerHTML = `
            <div class="astro-info-item">
                <div class="astro-info-label">Render Time</div>
                <div class="astro-info-value">${renderTime.toFixed(2)}s</div>
            </div>
            <div class="astro-info-item">
                <div class="astro-info-label">Resolution</div>
                <div class="astro-info-value">${resolution || 'N/A'}</div>
            </div>
            <div class="astro-info-item">
                <div class="astro-info-label">Engine</div>
                <div class="astro-info-value">${engine}</div>
            </div>
            <div class="astro-info-item">
                <div class="astro-info-label">Samples</div>
                <div class="astro-info-value">${samples}</div>
            </div>
        `;
    };

    updateInfo();
    model.on('change:render_time', updateInfo);
    model.on('change:resolution', updateInfo);
    model.on('change:render_engine', updateInfo);
    model.on('change:samples', updateInfo);

    container.appendChild(infoPanel);

    // Ops Editor Functions
    function loadOpsEditor() {
        // Request ops data from Python backend
        model.set('request_ops_data', model.get('request_ops_data') + 1);
        model.save_changes();
    }

    function createOpsUI(opsData) {
        opsEditorContent.innerHTML = '';

        if (!opsData || Object.keys(opsData).length === 0) {
            opsEditorContent.innerHTML = '<div class="astro-placeholder">🔍 Loading Blender ops...</div>';
            return;
        }

        // Create ops categories
        Object.keys(opsData).forEach(category => {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'astro-ops-category';

            const categoryHeader = document.createElement('div');
            categoryHeader.className = 'astro-ops-category-header';
            categoryHeader.textContent = `📁 ${category}`;
            categoryHeader.onclick = () => {
                const content = categoryDiv.querySelector('.astro-ops-category-content');
                content.style.display = content.style.display === 'none' ? 'block' : 'none';
            };
            categoryDiv.appendChild(categoryHeader);

            const categoryContent = document.createElement('div');
            categoryContent.className = 'astro-ops-category-content';
            categoryContent.style.display = 'none';

            // Add ops in this category
            opsData[category].forEach(op => {
                const opDiv = document.createElement('div');
                opDiv.className = 'astro-ops-item';

                const opHeader = document.createElement('div');
                opHeader.className = 'astro-ops-item-header';
                opHeader.textContent = `⚙️ ${op.name}`;
                opHeader.onclick = () => {
                    const params = opDiv.querySelector('.astro-ops-params');
                    params.style.display = params.style.display === 'none' ? 'block' : 'none';
                };
                opDiv.appendChild(opHeader);

                const paramsDiv = document.createElement('div');
                paramsDiv.className = 'astro-ops-params';
                paramsDiv.style.display = 'none';

                // Add parameter inputs
                if (op.parameters && op.parameters.length > 0) {
                    op.parameters.forEach(param => {
                        const paramDiv = document.createElement('div');
                        paramDiv.className = 'astro-ops-param';

                        const label = document.createElement('label');
                        label.textContent = param.name;
                        label.className = 'astro-ops-param-label';
                        paramDiv.appendChild(label);

                        const input = createParameterInput(param);
                        input.className = 'astro-ops-param-input';
                        paramDiv.appendChild(input);

                        if (param.description) {
                            const desc = document.createElement('div');
                            desc.className = 'astro-ops-param-desc';
                            desc.textContent = param.description;
                            paramDiv.appendChild(desc);
                        }

                        paramsDiv.appendChild(paramDiv);
                    });

                    // Execute button
                    const executeBtn = document.createElement('button');
                    executeBtn.className = 'astro-ops-execute-btn';
                    executeBtn.textContent = '▶️ Execute';
                    executeBtn.onclick = () => executeOp(category, op.name, paramsDiv);
                    paramsDiv.appendChild(executeBtn);
                }

                opDiv.appendChild(paramsDiv);
                categoryContent.appendChild(opDiv);
            });

            categoryDiv.appendChild(categoryContent);
            opsEditorContent.appendChild(categoryDiv);
        });
    }

    function createParameterInput(param) {
        let input;

        switch (param.type) {
            case 'boolean':
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = param.default || false;
                break;
            case 'int':
                input = document.createElement('input');
                input.type = 'number';
                input.step = '1';
                input.value = param.default || 0;
                if (param.min !== undefined) input.min = param.min;
                if (param.max !== undefined) input.max = param.max;
                break;
            case 'float':
                input = document.createElement('input');
                input.type = 'number';
                input.step = '0.01';
                input.value = param.default || 0.0;
                if (param.min !== undefined) input.min = param.min;
                if (param.max !== undefined) input.max = param.max;
                break;
            case 'string':
                input = document.createElement('input');
                input.type = 'text';
                input.value = param.default || '';
                break;
            case 'enum':
                input = document.createElement('select');
                if (param.items) {
                    param.items.forEach(item => {
                        const option = document.createElement('option');
                        option.value = item.identifier;
                        option.textContent = item.name;
                        if (item.identifier === param.default) option.selected = true;
                        input.appendChild(option);
                    });
                }
                break;
            case 'vector':
                input = document.createElement('div');
                input.className = 'astro-ops-vector-input';
                const dimensions = param.size || 3;
                for (let i = 0; i < dimensions; i++) {
                    const vectorInput = document.createElement('input');
                    vectorInput.type = 'number';
                    vectorInput.step = '0.01';
                    vectorInput.value = (param.default && param.default[i]) || 0;
                    vectorInput.placeholder = ['X', 'Y', 'Z'][i] || `${i}`;
                    input.appendChild(vectorInput);
                }
                break;
            default:
                input = document.createElement('input');
                input.type = 'text';
                input.value = param.default || '';
        }

        return input;
    }

    function executeOp(category, opName, paramsDiv) {
        const params = {};
        const paramInputs = paramsDiv.querySelectorAll('.astro-ops-param');

        paramInputs.forEach(paramDiv => {
            const label = paramDiv.querySelector('.astro-ops-param-label').textContent;
            const input = paramDiv.querySelector('.astro-ops-param-input');

            if (input.type === 'checkbox') {
                params[label] = input.checked;
            } else if (input.type === 'number') {
                params[label] = parseFloat(input.value) || 0;
            } else if (input.tagName === 'SELECT') {
                params[label] = input.value;
            } else if (input.className.includes('vector-input')) {
                const vectorInputs = input.querySelectorAll('input');
                params[label] = Array.from(vectorInputs).map(vi => parseFloat(vi.value) || 0);
            } else {
                params[label] = input.value;
            }
        });

        // Send execution request to backend
        model.set('execute_op', JSON.stringify({
            category: category,
            operation: opName,
            parameters: params
        }));
        model.save_changes();
    }

    // Listen for ops data updates
    model.on('change:ops_data', () => {
        const opsData = model.get('ops_data');
        if (opsData) {
            try {
                const parsedData = JSON.parse(opsData);
                createOpsUI(parsedData);
            } catch (e) {
                console.error('Failed to parse ops data:', e);
            }
        }
    });

    el.appendChild(container);
}

export default { render }; 