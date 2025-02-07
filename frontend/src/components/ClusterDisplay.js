// src/components/ClusterDisplay.js

import React, { useState } from 'react';
import Modal from './Modal';

const ClusterDisplay = ({ clusters }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedQuestions, setSelectedQuestions] = useState([]);

  const handleOpenModal = (questions) => {
    setSelectedQuestions(questions);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedQuestions([]); // Clear selected questions when closing
  };

  return (
    <div className="container mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Clusters</h2>
      {Object.entries(clusters).length === 0 ? (
        <p className="text-gray-500">No clusters available</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(clusters).map(([label, data]) => (
            <div key={label} className="border border-gray-300 rounded-lg p-4 shadow-md hover:shadow-lg transition duration-200">
              <h3 className="text-lg font-semibold">Cluster {label}</h3>
              <p className="mt-2">
                <strong>Representative Question:</strong> {data.representative_question}
              </p>
              <p className="mt-1">
                <strong>Number of Questions:</strong> {data.questions.length}
              </p>
              <button
                onClick={() => handleOpenModal(data.questions)}
                className="mt-4 bg-blue-600 text-white font-semibold py-2 px-4 rounded hover:bg-blue-700 transition duration-200"
              >
                View Questions
              </button>
            </div>
          ))}
        </div>
      )}
      <Modal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        questions={selectedQuestions}
      />
    </div>
  );
};

export default ClusterDisplay;
