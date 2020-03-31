import React from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import "./Wine.css";

function Wine({ id, year, title, summary, poster, genres }) {
  return (
    <div className="wine">
      <Link
        to={{
          pathname: `/wine/${id}`,
          state: {
            year,
            title,
            summary,
            poster,
            genres
          }
        }}
      >
        <img src={poster} alt={title} title={title} />
        <div className="wine__data">
          <h3 className="wine__title">{title}</h3>
          <h5 className="wine__year">{year}</h5>
          <ul className="wine__genres">
            {genres.map((genre, index) => (
              <li key={index} className="genres__genre">
                {genre}
              </li>
            ))}
          </ul>
          <p className="wine__summary">{summary.slice(0, 180)}...</p>
        </div>
      </Link>
    </div>
  );
}

Wine.propTypes = {
  id: PropTypes.number.isRequired,
  year: PropTypes.number.isRequired,
  title: PropTypes.string.isRequired,
  summary: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired,
  genres: PropTypes.arrayOf(PropTypes.string).isRequired
};

export default Wine;
