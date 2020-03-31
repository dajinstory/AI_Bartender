import React from "react";
import { HashRouter, Route } from "react-router-dom";
import Home from "./routes/Home";
import About from "./routes/About";
import Detail from "./routes/Detail";
import ImageSearch from "./routes/ImageSearch";
import Navigation from "./components/Navigation";
import "./App.css";

function App() {
  return (
    <HashRouter>
      <Navigation />
      <Route path="/" exact={true} component={Home} />
      <Route path="/about" component={About} />
      <Route path="/image_search" component={ImageSearch} />
      <Route path="/wine/:id" component={Detail} />
    </HashRouter>
  );
}

export default App;
