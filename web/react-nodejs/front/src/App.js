// import libraries
import React from "react";
import { HashRouter, Route } from "react-router-dom";

// main pages, components, and css file
import Home from "./routes/Home";
import About from "./routes/About";
import Detail from "./routes/Detail";
import ImageSearch from "./routes/ImageSearch";
import Navigation from "./components/Navigation";
import "./App.css";

// prototype pages
import ProtoType from "./routes/ProtoType";

function App() {
  return (
    <HashRouter>
      <Navigation />
      <Route path="/" exact={true} component={Home} />
      <Route path="/about" component={About} />
      <Route path="/image_search" component={ImageSearch} />
      <Route path="/wine/:id" component={Detail} />
      <Route path="/prototype" component={ProtoType} />
    </HashRouter>
  );
}

export default App;
