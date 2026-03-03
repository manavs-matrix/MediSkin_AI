// Handle image preview on file selection
document.getElementById('imageInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    
    if (file) {
        // Read the file and display preview
        const reader = new FileReader();
        
        reader.onload = function(e) {
            // Update the preview image source
            document.getElementById('previewImg').src = e.target.result;
            
            // Update the filename display
            document.getElementById('fileName').textContent = file.name;
            
            // Show the preview container
            document.getElementById('imagePreview').classList.remove('d-none');
        };
        
        reader.readAsDataURL(file);
    }
});

// Handle drag and drop functionality
const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop zone when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('drag-over');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('drag-over');
    }, false);
});

// Handle dropped files
dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    imageInput.files = files;
    
    // Trigger change event to show preview
    const event = new Event('change', { bubbles: true });
    imageInput.dispatchEvent(event);
}, false);

// Handle click on drop zone to open file dialog
dropZone.addEventListener('click', (e) => {
    // Don't trigger if clicking on the button
    if (!e.target.closest('label')) {
        imageInput.click();
    }
});
