'use client';

import React, { useRef, useEffect, useState } from 'react';
import { BrowserMultiFormatReader } from '@zxing/library';
import { Camera, X, Upload } from 'lucide-react';

interface QRScannerProps {
  onScan: (result: string) => void;
  onError?: (error: string) => void;
  onClose: () => void;
}

export const QRScanner: React.FC<QRScannerProps> = ({ onScan, onError, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState<string>('');
  const [codeReader, setCodeReader] = useState<BrowserMultiFormatReader | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const reader = new BrowserMultiFormatReader();
    setCodeReader(reader);

    return () => {
      reader.reset();
    };
  }, []);

  const startScanning = async () => {
    if (!codeReader || !videoRef.current) return;

    try {
      setIsScanning(true);
      setError('');

      const result = await codeReader.decodeOnceFromVideoDevice(undefined, videoRef.current);
      
      if (result) {
        onScan(result.getText());
        stopScanning();
      }
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to scan QR code';
      setError(errorMessage);
      onError?.(errorMessage);
      setIsScanning(false);
    }
  };

  const stopScanning = () => {
    if (codeReader) {
      codeReader.reset();
    }
    setIsScanning(false);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !codeReader) return;

    try {
      setError('');
      
      // Create an image element to decode from
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      img.onload = async () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx?.drawImage(img, 0, 0);
        
        try {
          const result = await codeReader.decodeFromImageElement(img);
          if (result) {
            onScan(result.getText());
          }
        } catch (err: any) {
          const errorMessage = 'Could not read QR code from image';
          setError(errorMessage);
          onError?.(errorMessage);
        }
      };
      
      img.src = URL.createObjectURL(file);
    } catch (err: any) {
      const errorMessage = 'Could not read QR code from image';
      setError(errorMessage);
      onError?.(errorMessage);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Scan QR Code</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <div className="space-y-4">
          {/* Camera View */}
          <div className="relative">
            <video
              ref={videoRef}
              className="w-full h-64 bg-gray-100 rounded-lg object-cover"
              autoPlay
              playsInline
              muted
            />
            
            {/* Scanning Overlay */}
            <div className="absolute inset-0 border-2 border-dashed border-blue-500 rounded-lg flex items-center justify-center">
              <div className="text-center text-white bg-black bg-opacity-50 p-2 rounded">
                <Camera className="h-8 w-8 mx-auto mb-2" />
                <p className="text-sm">Position QR code in frame</p>
              </div>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          )}

          {/* Controls */}
          <div className="flex space-x-2">
            <button
              onClick={isScanning ? stopScanning : startScanning}
              disabled={!codeReader}
              className={`flex-1 py-2 px-4 rounded-lg font-medium ${
                isScanning
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {isScanning ? 'Stop Scanning' : 'Start Camera'}
            </button>

            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload
            </button>
          </div>

          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          {/* Instructions */}
          <div className="text-sm text-gray-600 space-y-1">
            <p>• Point your camera at a QR code</p>
            <p>• Or upload an image containing a QR code</p>
            <p>• Make sure the QR code is clearly visible</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QRScanner;