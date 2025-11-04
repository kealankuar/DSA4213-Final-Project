document.addEventListener('DOMContentLoaded', () => {
    
    const resumeForm = document.getElementById('resume-form');
    const fileInput = document.getElementById('resume-file');
    const queryInput = document.getElementById('query-text'); 
    const submitButton = document.getElementById('submit-btn');
    
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

    resumeForm.addEventListener('submit', async (event) => {
        event.preventDefault(); 

        const resumeFile = fileInput.files[0];
        const query = queryInput.value;

        if (!resumeFile) {
            showToast("Please input your resume!", "error")
            return;
        }
        if (!query.trim()) {
            showToast("Please input your query!", "error")
            queryInput.focus();
            return;
        }

        resultContainer.classList.remove('hidden');
        loadingSpinner.classList.remove('hidden');
        responseArea.classList.add('hidden');
        submitButton.disabled = true;
        submitButton.textContent = 'Analyzing...';

        //hardcode answers here

        setTimeout(() => {
            
            const coursesAnswer = "Suggested Courses to Improve:\n" +
                                  "1.  Advanced Excel for Marketing: To boost your modeling speed.\n" +
                                  "2.  Python for Marketing Analysis: To automate tasks and align with modern roles.\n";

            const defaultAnswer = "Based on your resume, you are a strong candidate for a Marketing Analyst role.\n\n" +
                                  "Strengths:\n" +
                                  "Strong quantitative background.\n" +
                                  "Experience with marketing campaigns (from your 'Project Sales' entry).\n";

            const improvementsQuery = "What are some courses that I can take to improve my skills?";
            
            let answerToShow = "";

            if (query === improvementsQuery) {
                answerToShow = coursesAnswer;
            } else {
                answerToShow = defaultAnswer;
            }

            answerText.textContent = answerToShow; 
            responseArea.classList.remove('hidden');

            loadingSpinner.classList.add('hidden');
            submitButton.disabled = false;
            submitButton.textContent = 'Analyze My Career';

        }, 1500); 
        
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