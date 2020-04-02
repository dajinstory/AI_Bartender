import React from "react";
import { Link } from "react-router-dom";
import "./Navigation.css";

function Navigation() {
  return (
    <div className="nav">
      <Link to="/">Home</Link>
      <Link to="/about">About</Link>
      <Link to="/image_search">Image_Search</Link>
      <Link to="/prototype">__prototypes__</Link>
    </div>
  );
}

export default Navigation;
