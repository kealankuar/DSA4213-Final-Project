console.log('✓ Script loaded!');

document.addEventListener('DOMContentLoaded', () => {
    console.log('✓ DOM loaded, initializing...');
    
    const resumeForm = document.getElementById('resume-form');
    const fileInput = document.getElementById('resume-file');
    const queryInput = document.getElementById('query-text'); 
    const submitButton = document.getElementById('submit-btn');
    
    console.log('✓ Form elements:', { resumeForm, fileInput, queryInput, submitButton });
    
    const dragDropLabel = document.querySelector('.drag-drop-label');
    const dragDropText = document.getElementById('drag-drop-text');
    const uploadIcon = document.querySelector('.drag-drop-content i');

    const resultContainer = document.getElementById('result-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const responseArea = document.getElementById('response-area');
    const answerText = document.getElementById('answer-text');

    const promptButtons = document.querySelectorAll('.prompt-btn');
    const copyBtn = document.getElementById('copy-btn');
    const askAnotherBtn = document.getElementById('ask-another-btn');

    const allowed_ext = ['.pdf', '.doc', '.docx', '.txt'];


    function resetFileInput() {
        dragDropText.innerHTML = '<strong>Click to upload</strong> or drag and drop';
        dragDropLabel.classList.remove('file-selected');
        uploadIcon.className = 'fas fa-cloud-upload-alt'; 
        fileInput.value = null; 
    }

    function updateLabelWithFileName(file) {
        dragDropText.innerHTML = `<strong>File:</strong> ${file.name}`;
        dragDropLabel.classList.add('file-selected');
        uploadIcon.className = 'fas fa-check-circle'; 
    }
    
    function isFileValid(file) {
        if (!file) return false;
        const fileExtension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
        return allowed_ext.includes(fileExtension);
    }

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length === 0) {
            resetFileInput();
            return;
        }
        const file = e.target.files[0];
        if (isFileValid(file)) {
            updateLabelWithFileName(file);
        } else {
            showToast('Invalid file type. Please select a .pdf, .doc, .docx, or .txt file.', 'error');
            resetFileInput();
        }
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dragDropLabel.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dragDropLabel.addEventListener(eventName, () => {
            dragDropLabel.classList.add('dragover');
        }, false);
    });

    dragDropLabel.addEventListener('dragleave', () => {
        dragDropLabel.classList.remove('dragover');
    }, false);

    dragDropLabel.addEventListener('drop', (e) => {
        dragDropLabel.classList.remove('dragover');
        const droppedFiles = e.dataTransfer.files;

        if (droppedFiles.length > 0) {
            const file = droppedFiles[0];
            if (isFileValid(file)) {
                fileInput.files = droppedFiles; 
                updateLabelWithFileName(file);
            } else {
                showToast('Invalid file type. Please drop a .pdf, .doc, .docx, or .txt file.', 'error');
                resetFileInput(); 
            }
        }
    }, false);

    function showToast(message, type = "success") {
        const container = document.getElementById("toast-container");
        
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            setTimeout(() => toast.remove(), 500);
        }, 3000);
    }

    console.log('✓ Attaching submit event listener...');
    
    resumeForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // MUST be first!
        console.log('✓ Form submitted! (preventDefault called)'); 

        const resumeFile = fileInput.files[0];
        const query = queryInput.value;

        console.log('Resume file:', resumeFile);
        console.log('Query:', query);

        if (!resumeFile) {
            console.log('✗ No resume file');
            showToast("Please input your resume!", "error")
            return;
        }
        if (!query.trim()) {
            console.log('✗ No query text');
            showToast("Please input your query!", "error")
            queryInput.focus();
            return;
        }
        
        console.log('✓ Validation passed, proceeding with request...');

        resultContainer.classList.remove('hidden');
        loadingSpinner.classList.remove('hidden');
        responseArea.classList.add('hidden');
        submitButton.disabled = true;
        submitButton.textContent = 'Analyzing...';

        try {
            // Create FormData to send file and query
            const formData = new FormData();
            formData.append('resume', resumeFile);
            formData.append('query', query);

            console.log('Sending request to backend...');

            // Send to backend API
            let response;
            try {
                response = await fetch('http://localhost:5000/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                console.log('✓ Fetch completed, response received');
            } catch (fetchError) {
                console.error('FETCH ERROR:', fetchError);
                throw fetchError;
            }

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to analyze');
            }

            const data = await response.json();
            console.log('✓ JSON parsed successfully');

            // Debug: Log the response
            console.log('Backend response:', data);
            console.log('Answer field:', data.answer);
            console.log('Answer element:', answerText);

            // Display the generated answer
            if (data.answer) {
                answerText.textContent = data.answer;
                console.log('Answer set successfully');
            } else {
                answerText.textContent = 'No answer generated';
                console.log('No answer in response');
            }
            
            // Show the response area
            console.log('Showing response area...');
            resultContainer.classList.remove('hidden');
            responseArea.classList.remove('hidden');
            loadingSpinner.classList.add('hidden');
            
            showToast("Analysis complete!", "success");

        } catch (error) {
            console.error('=== ERROR CAUGHT ===');
            console.error('Error object:', error);
            console.error('Error name:', error.name);
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
            console.error('===================');
            showToast(`Error: ${error.message}`, "error");
            answerText.textContent = `Error processing your request: ${error.message}`;
            responseArea.classList.remove('hidden');
        } finally {
            loadingSpinner.classList.add('hidden');
            submitButton.disabled = false;
            submitButton.textContent = 'Analyze My Career';
        }
    });

    promptButtons.forEach(button => {
        button.addEventListener('click', () => {
            const query = button.getAttribute('data-query');
            queryInput.value = query; 
            queryInput.focus(); 
        });
    });

    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(answerText.textContent)
            .then(() => {
                const originalIcon = copyBtn.innerHTML;
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                showToast("Copied to clipboard!", "success"); 
                setTimeout(() => {
                    copyBtn.innerHTML = originalIcon;
                }, 1500);
            })
            .catch(err => {
                console.error('Failed to copy text: ', err);
                showToast("Failed to copy!", "error");
            });
    });

    askAnotherBtn.addEventListener('click', () => {
        queryInput.value = ''; 
        queryInput.focus(); 
        
        queryInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

});