var thrift = require('thrift');
var Bartender = require('../thrift_modules/gen-nodejs/Bartender');
var ttypes = require('../thrift_modules/gen-nodejs/bartender_api_types');
const assert = require('assert');

var transport = thrift.TBufferedTransport;
var protocol = thrift.TBinaryProtocol;

var connection = thrift.createConnection("localhost", 10101, {
    transport : transport,
    protocol : protocol
});

connection.on('error', function(err) {
    assert(false, err);
});

// Create a Calculator client with the connection
var client = thrift.createClient(Bartender, connection);


client.test_function("test_input",function(err, response) {
    if (err){
        console.log("error : " + err)
    } else{
        console.log('ping() ' + response);
    }

});