// src/App.js

import React, { useState } from 'react';
import QuestionInput from './components/QuestionInput';
import ClusterDisplay from './components/ClusterDisplay';
import './App.css';

const App = () => {
  const [clusters, setClusters] = useState({});

  const handleClustersReceived = (data) => {
    setClusters(data.clusters);
  };

  return (
    <div className="App">
      <QuestionInput onClustersReceived={handleClustersReceived} />
      <ClusterDisplay clusters={clusters} />
    </div>
  );
};

export default App;
