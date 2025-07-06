document.addEventListener('DOMContentLoaded', function() {
    // API endpoint - backend URL
    const API_URL = 'http://localhost:5000/api';
    
    // Check if the API is available
    fetch(`${API_URL.split('/api')[0]}/health`)
        .then(response => {
            if (response.ok) {
                console.log('API server is available');
                return response.json();
            } else {
                throw new Error('API server returned an error');
            }
        })
        .then(data => {
            console.log('API health check:', data);
        })
        .catch(error => {
            console.error('API server is not available:', error);
            showApiError();
        });
    
    function showApiError() {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger text-center';
        errorDiv.style.margin = '20px';
        errorDiv.innerHTML = `
            <h4>Cannot connect to API server</h4>
            <p>Make sure the backend server is running at ${API_URL.split('/api')[0]}</p>
            <p>You can start it by running: <code>python app.py</code></p>
        `;
        document.querySelector('.container').prepend(errorDiv);
    }
    
    // Face Analysis Variables
    const faceStaticRadio = document.getElementById('faceStatic');
    const faceRealtimeRadio = document.getElementById('faceRealtime');
    const faceStaticInput = document.getElementById('faceStaticInput');
    const faceRealtimeInput = document.getElementById('faceRealtimeInput');
    const faceImageUpload = document.getElementById('faceImageUpload');
    const analyzeFaceBtn = document.getElementById('analyzeFaceBtn');
    const faceResultContent = document.getElementById('faceResultContent');
    
    // Webcam Variables
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopWebcamBtn = document.getElementById('stopWebcamBtn');
    const captureFrameBtn = document.getElementById('captureFrameBtn');
    const webcamVideo = document.getElementById('webcamVideo');
    const captureCanvas = document.getElementById('captureCanvas');
    let webcamStream = null;
    
    // Voice Analysis Variables
    const voiceUploadRadio = document.getElementById('voiceUpload');
    const voiceRecordRadio = document.getElementById('voiceRecord');
    const voiceUploadInput = document.getElementById('voiceUploadInput');
    const voiceRecordInput = document.getElementById('voiceRecordInput');
    const voiceAudioUpload = document.getElementById('voiceAudioUpload');
    const analyzeVoiceBtn = document.getElementById('analyzeVoiceBtn');
    const voiceResultContent = document.getElementById('voiceResultContent');
    
    // Voice Recording Variables
    const startRecordingBtn = document.getElementById('startRecordingBtn');
    const stopRecordingBtn = document.getElementById('stopRecordingBtn');
    const analyzeRecordingBtn = document.getElementById('analyzeRecordingBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const audioPlayback = document.getElementById('audioPlayback');
    let mediaRecorder = null;
    let audioChunks = [];
    let audioBlob = null;
    
    // Handwriting Analysis Variables
    const handwritingImageUpload = document.getElementById('handwritingImageUpload');
    const analyzeHandwritingBtn = document.getElementById('analyzeHandwritingBtn');
    const handwritingResultContent = document.getElementById('handwritingResultContent');
    
    // Face Input Type Toggle
    faceStaticRadio.addEventListener('change', function() {
        if (this.checked) {
            faceStaticInput.style.display = 'block';
            faceRealtimeInput.style.display = 'none';
            stopWebcam();
        }
    });
    
    faceRealtimeRadio.addEventListener('change', function() {
        if (this.checked) {
            faceStaticInput.style.display = 'none';
            faceRealtimeInput.style.display = 'block';
        }
    });
    
    // Voice Input Type Toggle
    voiceUploadRadio.addEventListener('change', function() {
        if (this.checked) {
            voiceUploadInput.style.display = 'block';
            voiceRecordInput.style.display = 'none';
            stopRecording();
        }
    });
    
    voiceRecordRadio.addEventListener('change', function() {
        if (this.checked) {
            voiceUploadInput.style.display = 'none';
            voiceRecordInput.style.display = 'block';
        }
    });
    
    // Face Image Upload Preview
    faceImageUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const previewImg = document.createElement('img');
            previewImg.className = 'preview-image';
            previewImg.file = this.files[0];
            
            const reader = new FileReader();
            reader.onload = (function(aImg) {
                return function(e) {
                    aImg.src = e.target.result;
                };
            })(previewImg);
            
            reader.readAsDataURL(this.files[0]);
            
            // Remove any existing preview
            const existingPreview = faceStaticInput.querySelector('.preview-image');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            faceStaticInput.appendChild(previewImg);
        }
    });
    
    // Handwriting Image Upload Preview
    handwritingImageUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const previewImg = document.createElement('img');
            previewImg.className = 'preview-image';
            previewImg.file = this.files[0];
            
            const reader = new FileReader();
            reader.onload = (function(aImg) {
                return function(e) {
                    aImg.src = e.target.result;
                };
            })(previewImg);
            
            reader.readAsDataURL(this.files[0]);
            
            // Remove any existing preview
            const existingPreview = document.querySelector('#handwriting .preview-image');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            document.querySelector('#handwriting .mb-3').appendChild(previewImg);
        }
    });
    
    // Webcam Functions
    startWebcamBtn.addEventListener('click', startWebcam);
    stopWebcamBtn.addEventListener('click', stopWebcam);
    captureFrameBtn.addEventListener('click', captureAndAnalyzeFrame);
    
    function startWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    webcamStream = stream;
                    webcamVideo.srcObject = stream;
                    webcamVideo.play();
                    webcamVideo.style.display = 'block';
                    startWebcamBtn.style.display = 'none';
                    stopWebcamBtn.style.display = 'inline-block';
                    captureFrameBtn.style.display = 'inline-block';
                })
                .catch(function(error) {
                    console.error("Error accessing webcam:", error);
                    showError("Could not access webcam. Please check permissions.", faceResultContent);
                });
        } else {
            showError("Your browser does not support webcam access.", faceResultContent);
        }
    }
    
    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
            webcamVideo.style.display = 'none';
            startWebcamBtn.style.display = 'inline-block';
            stopWebcamBtn.style.display = 'none';
            captureFrameBtn.style.display = 'none';
        }
    }
    
    function captureAndAnalyzeFrame() {
        if (webcamStream) {
            const context = captureCanvas.getContext('2d');
            captureCanvas.width = webcamVideo.videoWidth;
            captureCanvas.height = webcamVideo.videoHeight;
            context.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
            
            captureCanvas.toBlob(function(blob) {
                analyzeFaceFromBlob(blob);
            }, 'image/jpeg');
        }
    }
    
    // Voice Recording Functions
    startRecordingBtn.addEventListener('click', startRecording);
    stopRecordingBtn.addEventListener('click', stopRecording);
    analyzeRecordingBtn.addEventListener('click', analyzeRecording);
    
    function startRecording() {
        audioChunks = [];
        
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(function(stream) {
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = function(e) {
                        audioChunks.push(e.data);
                    };
                    
                    mediaRecorder.onstop = function() {
                        audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        audioPlayback.src = audioUrl;
                        audioPlayback.style.display = 'block';
                        
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    mediaRecorder.start();
                    recordingStatus.textContent = "Recording... Speak now";
                    recordingStatus.classList.add('recording-active');
                    startRecordingBtn.style.display = 'none';
                    stopRecordingBtn.style.display = 'inline-block';
                })
                .catch(function(error) {
                    console.error("Error accessing microphone:", error);
                    showError("Could not access microphone. Please check permissions.", voiceResultContent);
                });
        } else {
            showError("Your browser does not support audio recording.", voiceResultContent);
        }
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            recordingStatus.textContent = "Recording stopped";
            recordingStatus.classList.remove('recording-active');
            stopRecordingBtn.style.display = 'none';
            analyzeRecordingBtn.style.display = 'inline-block';
        }
    }
    
    function analyzeRecording() {
        if (audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            
            showLoading(voiceResultContent);
            
            fetch(`${API_URL}/voice`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayVoiceResults(data);
            })
            .catch(error => {
                console.error("Error analyzing voice:", error);
                showError("Error analyzing voice recording.", voiceResultContent);
            });
        }
    }
    
    // Analysis Button Event Listeners
    analyzeFaceBtn.addEventListener('click', function() {
        if (faceImageUpload.files && faceImageUpload.files[0]) {
            const formData = new FormData();
            formData.append('image', faceImageUpload.files[0]);
            
            showLoading(faceResultContent);
            
            fetch(`${API_URL}/face/static`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayFaceResults(data);
            })
            .catch(error => {
                console.error("Error analyzing face:", error);
                showError("Error analyzing face image.", faceResultContent);
            });
        } else {
            showError("Please select an image first.", faceResultContent);
        }
    });
    
    analyzeVoiceBtn.addEventListener('click', function() {
        if (voiceAudioUpload.files && voiceAudioUpload.files[0]) {
            const formData = new FormData();
            formData.append('audio', voiceAudioUpload.files[0]);
            
            showLoading(voiceResultContent);
            
            fetch(`${API_URL}/voice`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayVoiceResults(data);
            })
            .catch(error => {
                console.error("Error analyzing voice:", error);
                showError("Error analyzing voice audio.", voiceResultContent);
            });
        } else {
            showError("Please select an audio file first.", voiceResultContent);
        }
    });
    
    analyzeHandwritingBtn.addEventListener('click', function() {
        if (handwritingImageUpload.files && handwritingImageUpload.files[0]) {
            const formData = new FormData();
            formData.append('image', handwritingImageUpload.files[0]);
            
            showLoading(handwritingResultContent);
            
            fetch(`${API_URL}/handwriting`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayHandwritingResults(data);
            })
            .catch(error => {
                console.error("Error analyzing handwriting:", error);
                showError("Error analyzing handwriting image.", handwritingResultContent);
            });
        } else {
            showError("Please select an image first.", handwritingResultContent);
        }
    });
    
    // Helper function to analyze face from blob (used for webcam)
    function analyzeFaceFromBlob(blob) {
        const formData = new FormData();
        formData.append('image', blob, 'webcam.jpg');
        
        showLoading(faceResultContent);
        
        fetch(`${API_URL}/face/static`, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayFaceResults(data);
        })
        .catch(error => {
            console.error("Error analyzing face:", error);
            showError("Error analyzing webcam image.", faceResultContent);
        });
    }
    
    // Result Display Functions
    function displayFaceResults(data) {
        if (data.error) {
            showError(data.error, faceResultContent);
            return;
        }
        
        let html = `
            <div class="mental-state">${data.mental_state}</div>
            <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
        `;
        
        if (data.features) {
            html += `<div class="features-list">
                <h5>Detected Features:</h5>
                <div class="feature-item">
                    <span>Eye Aspect Ratio:</span>
                    <span>${data.features.eye_aspect_ratio.toFixed(4)}</span>
                </div>
                <div class="feature-item">
                    <span>Brow Drop:</span>
                    <span>${data.features.brow_drop.toFixed(4)}</span>
                </div>
                <div class="feature-item">
                    <span>Lip Tightness:</span>
                    <span>${data.features.lip_tightness.toFixed(4)}</span>
                </div>
                <div class="feature-item">
                    <span>Blink Count:</span>
                    <span>${data.features.blink_count}</span>
                </div>
            </div>`;
        }
        
        faceResultContent.innerHTML = html;
    }
    
    function displayVoiceResults(data) {
        if (data.error) {
            showError(data.error, voiceResultContent);
            return;
        }
        
        let html = `
            <div class="mental-state">${data.emotion}</div>
            <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
        `;
        
        voiceResultContent.innerHTML = html;
    }
    
    function displayHandwritingResults(data) {
        if (data.error) {
            showError(data.error, handwritingResultContent);
            return;
        }
        
        let html = `
            <div class="mental-state">${data.emotion}</div>
            <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
        `;
        
        handwritingResultContent.innerHTML = html;
    }
    
    // Utility Functions
    function showError(message, container) {
        container.innerHTML = `<div class="error-message">${message}</div>`;
    }
    
    function showLoading(container) {
        container.innerHTML = `<div class="text-center">
            <p>Analyzing... Please wait</p>
            <div class="loading"></div>
        </div>`;
    }
}); 