import React from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import "./Object.css";

function Object({ r, c, len_r, len_c, poster }) {
  return (
      <div className="object">
          <img src={poster ? (poster):('/images/default.png')} />
          <div className="object__data">
            <h5>{r}</h5>
            <h5>{c}</h5>
            <h5>{len_r}</h5>
            <h5>{len_c}</h5>
          </div>
      </div>
  );
}

Object.propTypes = {
  r: PropTypes.number.isRequired,
  c: PropTypes.number.isRequired,
  len_r: PropTypes.number.isRequired,
  len_c: PropTypes.number.isRequired,
  poster: PropTypes.string.isRequired
};

export default Object;
