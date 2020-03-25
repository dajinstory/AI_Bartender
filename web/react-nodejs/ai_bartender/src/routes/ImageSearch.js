import "./ImageSearch.css";
import React, { Component } from 'react'
import axios from 'axios';

class ImageSearch extends Component {
  constructor(props){
    super(props);
    this.state = {
      file: null
    };
    this.onFormSubmit = this.onFormSubmit.bind(this);
    this.onChange = this.onChange.bind(this);
  }

  onFormSubmit(e) {
      // preventdefault action
      e.preventDefault();

      // set params
      const formData = new FormData();
      formData.append('myImage', this.state.file);
      const config = {
          headers: {
              'content-type': 'multipart/form-data'
          }
      };

      // call post
      axios.post("http://localhost:8000/upload", formData, config)
          .then((response) => {
              alert("successfully uploaded");
          }).catch((error) => {
              alert("fail to upload image");
          }
      );
  }

  onChange(e){
      this.setState({file:e.target.files[0]})
  }

  render() {
    return (
        ///
        <div className="image_search__container">
          <form onSubmit={this.onFormSubmit}>
            <h2> File Upload </h2>
            <input type="file" name="selected_image" onChange={this.onChange}/>
            <button type="submit">Upload</button>
          </form>
        </div>
    )
  }
}

export default ImageSearch