import React from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import "./Object.css";

function Wine({ r, c, len_r, len_c, poster }) {
  return (
      <div className="wine">
          <img src={poster?(poster):('/images/default.png')} />
          <div className="wine__data">
            <h5>{r}</h5>
            <h5>{c}</h5>
            <h5>{len_r}</h5>
            <h5>{len_c}</h5>
          </div>
      </div>
  );
}

Wine.propTypes = {
  id: PropTypes.number.isRequired,
  year: PropTypes.number.isRequired,
  title: PropTypes.string.isRequired,
  summary: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired
};

export default Wine;
