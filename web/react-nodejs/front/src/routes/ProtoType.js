import React from 'react'
import axios from 'axios';
import Object from "../components/Object";
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
      objects: [],
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
          this.setState({ result: "uploaded", filename: response['data']['filename']});
          alert("Successfully uploaded\n");
        }).catch((error) => {
          alert("Fail to upload images\n" + error);
        }
    );
  }


  // detect objects in uploaded image
  onFormDetect = async(e) => {
    // prevent default action
    e.preventDefault();

    // get - detect objects
    axios.get("http://localhost:11000/detect", {params: {filename: this.state.filename}})
        .then((response) => {
          alert("Successfully detected");
          this.setState({ objects:response["data"]["objects"], result: "detected" });
        }).catch((error) => {
          alert("Fail to detect objects\n" + error);
        }
    );
  }


  // vectorize objects in uploaded image
  onFormVectorize = async(e) => {
    // prevent default action
    e.preventDefault();

    // get vectors
    axios.get("http://localhost:11000/vectorize", {params: {filename: this.state.filename}})
        .then((response) => {
          alert("Successfully vectorized");
          this.setState({ objects: response["data"]["objects"], result: "vectorized" });
        }).catch((error) => {
          alert("Fail to vectorize objects\n" + error);
        }
    );
  }

  // classify objects
  onFormClassify = async(e) => {
    // prevent default action
    e.preventDefault();

    // get - classify objects
    axios.get("http://localhost:11000/classify", {params: {filename: this.state.filename}})
        .then((response) => {
          alert("Successfully classified");
          this.setState({ objects: response["data"]["objects"], result: "classified" });
        }).catch((error) => {
          alert("Fail to classify objects\n" + error);
        }
    );
  }

  // rendering
  render() {
    const { file, filename, fileURL, result, objects} = this.state;
    return (
        <section className = "container">
          <div className="upload__container">
            <form onSubmit={this.onFormUpload}>
              <h3> Search objects</h3>
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

          {
            result=='classified ' ?(
                <div className="objects">
                  {objects.map(object => (
                      <Object
                          r={object.r ? (object.r):(9999)}
                          c={object.c ? (object.c):(9999)}
                          len_r={object.len_r ? (object.len_r):(9999)}
                          len_c={object.len_c ? (object.len_c):(9999)}
                          poster={object.URL ? (object.URL):(null)}
                      />
                  ))}
                </div>
            ) : (
                <div className="objects">
                  {objects.map(object => (
                      <Object
                          r={object.r ? (object.r):(9999)}
                          c={object.c ? (object.c):(9999)}
                          len_r={object.len_r ? (object.len_r):(9999)}
                          len_c={object.len_c ? (object.len_c):(9999)}
                          vector={object.vector ? (object.vector):([9999,9999,9999,9999])}
                          label={object.label ? (object.label):(9999)}
                          poster={object.URL ? (object.URL):(null)}
                      />
                  ))}
                </div>
            )
          }
        </section>
    )
  }
}

export default ProtoType

