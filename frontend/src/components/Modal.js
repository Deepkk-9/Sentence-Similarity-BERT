// src/components/Modal.js

import React from 'react';

const Modal = ({ isOpen, onClose, questions }) => {
  if (!isOpen) return null; // Don't render anything if not open

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-70 transition-opacity duration-300">
      <div className="bg-white rounded-lg shadow-lg p-6 max-w-lg w-full transform transition-transform duration-300 scale-100">
        <h2 className="text-2xl font-semibold mb-4 text-center">Questions in Cluster</h2>
        <ul className="list-disc pl-5 mb-4 h-96 overflow-scroll">
          {questions.map((question, index) => (
            <li key={index} className="mb-2 text-gray-800">{question}</li>
          ))}
        </ul>
        <div className="flex justify-center">
          <button
            onClick={onClose}
            className="bg-blue-600 text-white font-semibold py-2 px-4 rounded hover:bg-blue-700 transition duration-200 focus:outline-none focus:ring-2 focus:ring-blue-400"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default Modal;
