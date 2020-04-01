import React, { Component } from 'react'
import axios from 'axios';
import Wine from "../components/Wine";
import "./ImageSearch.css";

class ImageSearch extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      file: null,
      isLoading: true,
      wines: [],
    };
    this.onFormSubmit = this.onFormSubmit.bind(this);
    this.onChange = this.onChange.bind(this);
  }

  onFormSubmit = async(e) => {
      // prevent default action
      e.preventDefault();

      // set params
      const formData = new FormData();
      formData.append('selected_image', this.state.file);
      const config = {
          headers: {
              'content-type': 'multipart/form-data'
          }
      };

      // call post
      const search_response = axios.post("http://localhost:8000/upload", formData, config)
          .then((response) => {
              //alert("successfully uploaded and success : " + String(response["data"]["wines"]));
              alert("successfully uploaded" + String(JSON.stringify(response)));
              this.setState({ wines: response["data"]["wines"], isLoading: false });
          }).catch((error) => {
              alert("fail to upload image" + error);
          }
      );
  }

  onChange(e){
      this.setState({file:e.target.files[0]})
  }

  render() {
    const { file, isLoading, wines} = this.state;
    return (
        <section className = "wines__container">
          {isLoading ? (
              <div className="image_search__container">
                <form onSubmit={this.onFormSubmit}>
                  <h2> File Upload </h2>
                  <input type="file" name="selected_image" onChange={this.onChange}/>
                  <button type="submit">Search</button>
                </form>
              </div>
          ) : (
              <div className="wines">
                {wines.map(wine => (
                    <Wine
                        key={wine.x}
                        id={wine.x}
                        year={wine.len_x}
                        title={'name'}
                        summary={'summary'}
                        poster={null}
                    />
                ))}
              </div>
          )}
        </section>
    )
  }
}

export default ImageSearch

