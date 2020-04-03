import React from 'react'
import axios from 'axios';
import Wine from "../components/Wine";
import "./ProtoType.css";
import {Link} from "react-router-dom";

class ProtoType extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      file: null,
      filename: null,
      fileURL: null,
      result: null,
      wines: [],
    };
    // upload image
    this.onChange = this.onChange.bind(this);
    this.onFormUpload = this.onFormUpload.bind(this);

    // prototypes
    this.onFormDetect = this.onFormDetect.bind(this);
    this.onFormVectorize = this.onFormVectorize.bind(this);
    this.onFormClassify = this.onFormClassify.bind(this);
  }


  // upload image to server
  onChange(e){
    this.setState({file:e.target.files[0], fileURL:URL.createObjectURL(e.target.files[0])})
  }
  onFormUpload = async(e) => {
    // prevent default action
    e.preventDefault();

    // set the content of FormData
    const formData = new FormData();
    formData.append('selected_image', this.state.file); //key and value
    const config = {
      headers: {
        'content-type': 'multipart/form-data'
      }
    };

    // post - upload image
    axios.post("http://localhost:11000/upload", formData, config)
        .then((response) => {
          //alert("Successfully uploaded\n" + String(JSON.stringify(response)));
          alert("Successfully uploaded\n");
          this.setState({ result: "uploaded", filename: response['data']['filename']});
        }).catch((error) => {
          alert("Fail to upload images\n" + error);
        }
    );
  }


  // detect wines in uploaded image
  onFormDetect = async(e) => {
    // prevent default action
    e.preventDefault();

    // get - detect wines
    axios.get("http://localhost:11000/detect", {params: {filename: this.state.filename}})
        .then((response) => {
          alert("Successfully detected");
          this.setState({ wines: response["data"]["wines"], result: "detected" });
        }).catch((error) => {
          alert("Fail to detect wines\n" + error);
        }
    );
  }

  // vectorize wines in uploaded image
  onFormVectorize = async(e) => {
    // prevent default action
    e.preventDefault();

    // get vectors
    axios.get("http://localhost:11000/vectorize", {params: {filename: this.state.filename}})
        .then((response) => {
          alert("Successfully vectorized");
          this.setState({ wines: response["data"]["wines"], result: "vectorized" });
        }).catch((error) => {
          alert("Fail to vectorize wines\n" + error);
        }
    );
  }

  // classify wines
  onFormClassify = async(e) => {
    // prevent default action
    e.preventDefault();

    // get - classify wines
    axios.get("http://localhost:11000/classify", {params: {filename: this.state.filename}})
        .then((response) => {
          alert("Successfully classified");
          this.setState({ wines: response["data"]["wines"], result: "classified" });
        }).catch((error) => {
          alert("Fail to classify wines\n" + error);
        }
    );
  }

  // rendering
  render() {
    const { file, filename, fileURL, result, wines} = this.state;
    return (
        <section className = "container">
          <div className="upload__container">
            <form onSubmit={this.onFormUpload}>
              <h3> Search Wines</h3>
              {
                file ?(
                    <div>selected_image</div>
                ) : (
                    <div>default_image</div>
                )
              }
              <input type="file" name="selected_image" onChange={this.onChange}/>
              <button type="upload">Upload</button>
            </form>
            <hr/>
            <button onClick={this.onFormDetect}>Detect</button>
            <button onClick={this.onFormVectorize}>Vectorize</button>
            <button onClick={this.onFormClassify}>Classify</button>
          </div>
          <img src={fileURL?(fileURL):('/images/default.png')}  width='30%' height='30%' max-width='600' max-height='600'  />
          <div className="result__container">
            <div className="wines">
              {wines.map(wine => (
                  <Wine
                      id={wine.label ? (wine.id):(4112)}
                      year={wine.year ? (wine.year):(4112)}
                      title={wine.name ? (wine.name):(wine.r + wine.c + wine.)}
                      summary={wine.summary ? (wine.summary):("NO DESCRIPTION")}
                      poster={wine.poster ? (wine.poster):(null)}
                  />
              ))}
            </div>
          </div>
        </section>
    )
  }
}

export default ProtoType

