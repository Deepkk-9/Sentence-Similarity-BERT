import React, { useState } from 'react';
import axios from 'axios';

const QuestionInput = ({ onClustersReceived }) => {
    const [questions, setQuestions] = useState('');
    const [source, setSource] = useState('dbscan'); // Default selection
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        const questionArray = questions.split('\n').filter(q => q.trim() !== '');

        try {
            const response = await axios.post(`https://sentence-similarity-bert-deepkk-9.up.railway.app/cluster/${source}`, {
                questions: questionArray,
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
                        placeholder='Enter one question per line'
                        rows="4"
                        required
                        className="form-control mt-1 block w-full border border-gray-300 rounded-md shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50 p-2"
                    />
                </div>

                {/* Dropdown for selecting clustering algorithm */}
                <div className="form-group mt-4">
                    <label htmlFor="algorithm" className="block text-lg font-semibold">Select Clustering Algorithm:</label>
                    <select
                        id="algorithm"
                        value={source}
                        onChange={(e) => setSource(e.target.value)}
                        className="form-control mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                    >
                        <option value="dbscan">DBSCAN</option>
                        <option value="hdbscan">HDBSCAN</option>
                        <option value="agglomerative">Agglomerative</option>
                        <option value="lda">LDA</option>
                    </select>
                </div>

                <button
                    type="submit"
                    className="mt-4 bg-blue-600 text-white font-semibold py-2 px-4 rounded hover:bg-blue-700 transition duration-200"
                >
                    Submit
                </button>
            </form>

            {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
    );
};

export default QuestionInput;