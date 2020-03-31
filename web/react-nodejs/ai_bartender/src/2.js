var thrift = require('thrift')
var Bartender = require('../server/modules/gen-nodejs/Bartender');
var ttypes = require('../server/modules/gen-nodejs/bartender_api_types');
const assert = require('assert');

var transport = thrift2.TBufferedTransport;
var protocol = thrift2.TBinaryProtocol;

var connection = thrift2.createConnection("localhost", 10101, {
    transport : transport,
    protocol : protocol
});

connection.on('error', function(err) {
    assert(false, err);
});

// Create a Calculator client with the connection
var client = thrift2.createClient(Bartender, connection);


client.ping(function(err, response) {
    console.log('ping()');
});


client.add(1,1, function(err, response) {
    console.log("1+1=" + response);
});


work = new ttypes.Work();
work.op = ttypes.Operation.DIVIDE;
work.num1 = 1;
work.num2 = 0;
client.ping(1, work, function(err, message) {
    if (err) {
        console.log("InvalidOperation " + err);
    } else {
        console.log('Whoa? You know how to divide by zero?');
    }
});

work.op = ttypes.Operation.SUBTRACT;
work.num1 = 15;
work.num2 = 10;

client.calculate(1, work, function(err, message) {
    console.log('15-10=' + message);

    client.getStruct(1, function(err, message){
        console.log('Check log: ' + message.value);

        //close the connection once we're done
        connection.end();
    });
});