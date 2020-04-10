import React from "react";
import axios from "axios";
import Wine from "../components/Wine";
import "./Home.css";

class Home extends React.Component {
  state = {
    isLoading: true,
    wines: []
  };
  getWines = async () => {
    const tmp = await axios.get(
        "https://yts-proxy.now.sh/list_movies.json?sort_by=rating"
    );
    const {
      data: {
        data: { movies }
      }
    } = await axios.get(
      "https://yts-proxy.now.sh/list_movies.json?sort_by=rating"
    );
    this.setState({ wines: movies, isLoading: false });
  };
  componentDidMount() {
    this.getWines();
  }
  render() {
    const { isLoading, wines } = this.state;
    return (
      <section className="container">
        {isLoading ? (
          <div className="loader">
            <span className="loader__text">Loading...</span>
          </div>
        ) : (
          <div className="wines">
            {wines.map(wine => (
              <Wine
                key={wine.id}
                id={wine.id}
                year={wine.year}
                title={wine.title}
                summary={wine.summary}
                poster={wine.medium_cover_image}
              />
            ))}
          </div>
        )}
      </section>
    );
  }
}

export default Home;
