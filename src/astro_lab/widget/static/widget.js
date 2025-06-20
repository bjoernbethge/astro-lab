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

    // Quick scene controls
    const sceneControls = document.createElement('div');
    sceneControls.className = 'astro-scene-controls';

    const quickButtons = [
        { text: 'Galaxy', action: 'galaxy' },
        { text: 'Solar System', action: 'solar_system' },
        { text: 'Reset Scene', action: 'reset' },
        { text: 'Quick Render', action: 'render' }
    ];

    quickButtons.forEach(btn => {
        const button = document.createElement('button');
        button.className = 'astro-quick-button';
        button.textContent = btn.text;
        button.onclick = () => {
            model.set('trigger_render', model.get('trigger_render') + 1);
            model.save_changes();
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
    el.appendChild(container);
}

export default { render }; 