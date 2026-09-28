
import { useRef, useState } from 'react';
import { Upload, FileText, X, CheckCircle, AlertCircle } from 'lucide-react';

function FileUpload({ onFileSelected }) {
  const inputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [uploadMessage, setUploadMessage] = useState('');
  const [uploading, setUploading] = useState(false);

  async function handleFile(file) {
    setError('');
    setUploadMessage('');

    if (!file) return;

    const extension = file.name.split('.').pop().toLowerCase();

    if (extension !== 'csv') {
      setError('Please select a CSV file.');
      setSelectedFile(null);
      onFileSelected?.(null);
      return;
    }

    if (file.size === 0) {
      setError('The selected file is empty.');
      setSelectedFile(null);
      onFileSelected?.(null);
      return;
    }

    setSelectedFile(file);
    setUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/analyze-upload', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || 'Signal upload failed.');
      }

      onFileSelected?.(result);
      setUploadMessage(
        `Loaded ${result.total_samples.toLocaleString()} ECG/PPG samples.`
      );
    } catch (uploadError) {
      setSelectedFile(null);
      onFileSelected?.(null);
      setError(
        uploadError instanceof TypeError
          ? 'Cannot reach the backend. Start it from the backend folder with: python -m uvicorn app.main:app --reload'
          : uploadError.message
      );
    } finally {
      setUploading(false);
    }
  }

  function handleChange(event) {
    const file = event.target.files[0];
    handleFile(file);
  }

  function handleDrop(event) {
    event.preventDefault();

    const file = event.dataTransfer.files[0];
    handleFile(file);
  }

  function handleRemove() {
    setSelectedFile(null);
    setError('');
    setUploadMessage('');

    if (inputRef.current) {
      inputRef.current.value = '';
    }

    if (onFileSelected) {
      onFileSelected(null);
    }
  }

  return (
    <div className="upload-panel">
      <div className="upload-heading">
        <div>
          <h3>Upload Signal Data</h3>
          <p>Select a CSV file containing ECG or PPG measurements.</p>
        </div>

        <div className="upload-heading-icon">
          <Upload size={20} />
        </div>
      </div>

      <div
        className="upload-dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <div className="upload-icon">
          <Upload size={25} />
        </div>

        <h4>{uploading ? 'Uploading and analyzing...' : 'Drop your CSV file here'}</h4>

        <p>
          or <span className="browse-text">browse files</span> from your
          computer
        </p>

        <span className="file-format">Supported format: .csv</span>

        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          onChange={handleChange}
          hidden
        />
      </div>

      {error && (
        <div className="upload-message upload-error">
          <AlertCircle size={17} />
          {error}
        </div>
      )}

      {uploadMessage && (
        <div className="upload-message upload-success">
          <CheckCircle size={17} />
          {uploadMessage}
        </div>
      )}

      {selectedFile && (
        <div className="selected-file">
          <div className="selected-file-icon">
            <FileText size={20} />
          </div>

          <div className="selected-file-info">
            <strong>{selectedFile.name}</strong>
            <span>
              {(selectedFile.size / 1024).toFixed(2)} KB
            </span>
          </div>

          <CheckCircle className="file-success" size={19} />

          <button
            className="remove-file-button"
            onClick={handleRemove}
            aria-label="Remove selected file"
          >
            <X size={17} />
          </button>
        </div>
      )}

      <p className="upload-disclaimer">
        File selection only. Signal processing and model prediction will
        be performed after backend integration.
      </p>
    </div>
  );
}

export default FileUpload;