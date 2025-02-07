// src/components/QuestionInput.js

import React, { useState } from 'react';
import axios from 'axios';

const QuestionInput = ({ onClustersReceived }) => {
  const [questions, setQuestions] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const questionArray = questions.split('\n').filter(q => q.trim() !== '');

    try {
      const response = await axios.post('http://127.0.0.1:5000/cluster', {
        questions: questionArray,  // Correctly sending questionArray
      });
      onClustersReceived(response.data);
    } catch (err) {
      setError('Error processing questions. Please try again.');
      console.error(err);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mt-5">Clustered Questions</h1>
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="form-group">
          <label htmlFor="questions" className="block text-lg font-semibold">Enter Questions:</label>
          <textarea
            id="questions"
            value={questions}
            onChange={(e) => setQuestions(e.target.value)}
            placeholder='Enter one Question per line'
            rows="4"
            required
            className="form-control mt-1 block w-full border border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50 p-2"
          />
        </div>
        <button
          type="submit"
          className="mt-4 bg-blue-600 text-white font-semibold py-2 px-4 rounded hover:bg-blue-700 transition duration-200"
        >
          Submit
        </button>
      </form>

      <div className="mt-5">
        {/* This part will display clusters if available */}
        {/* You can replace this comment with a conditional rendering logic based on your application state */}
        {/* Example: {clusters.length > 0 && <ClusterDisplay clusters={clusters} />} */}
      </div>

      {error && <p className="text-red-500 mt-2">{error}</p>}
    </div>
  );
};

export default QuestionInput;
