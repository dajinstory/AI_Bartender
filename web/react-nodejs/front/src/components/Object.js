import React from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import "./Object.css";

function Object({ r, c, len_r, len_c, poster }) {
  return (
      <div className="object">
        <Link
            to={{
              pathname: `/object/${String(r)+'_'+String(c)}`,
              state: {
                r,
                c,
                len_r,
                len_c,
                poster
              }
            }}
        >
          <img src={poster ? (poster):('/images/default.png')} />
          <div className="object__data">
            <h3 className="object__name">UNKNOWN</h3>
            <h5 className="object__position">{r}, {c}, {len_r}, {len_c}</h5>
            <p className="object__summary">...</p>
          </div>
        </Link>
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
