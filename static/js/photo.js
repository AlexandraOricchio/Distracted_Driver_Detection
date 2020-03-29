var predictionNumber = data["Prediction"][1];

if (data['Success']) {
  switch(predictionNumber) {
      case '0':
        prediction = "This Driver is being Safe!";
        break;
      case '1': 
        prediction = "The Driver is texting with their Right Hand";
        break;
      case '2': 
        prediction = "The Driver is talking on the phone with their Right Hand";
        break;
      case '3': 
        prediction = "The Driver is texting with their Left Hand";
        break;
      case '4': 
        prediction = "The Driver is talking on the phone with their Left Hand";
        break;
      case '5': 
        prediction = "The Driver is Operating the Radio";
        break;
      case '6': 
        prediction = "The Driver is Drinking";
        break;
      case '7': 
        prediction = "The Driver is Reaching Behind";
        break;
      case '8': 
        prediction = "The Driver is Focused on their Hair and Makeup";
        break;
      case '9': 
        prediction = "The Driver is Talking to a Passenger";
        break;
        
      
      default: 
        prediction = "Something went wrong!"; }
  console.log(prediction);
  document.getElementById("prediction").innerHTML=prediction;

  console.log(data['Values'])
  var graph_array=data['Values'][0]
  graph_array.push(data['Values'][0][0])
  console.log(graph_array)

  var graph_data = [{
    type: 'scatterpolar',
    r: graph_array,
    theta: ['0','1','2','3','4','5','6','7','8','9','0'],
    fill: 'toself'
  }]
  var layout = {
    polar: {
      radialaxis: {
        visible: true,
        range: [0,1]
      }
    },
    showlegend: false
  }
  Plotly.newPlot("graph",graph_data,layout)
}
else{
  document.getElementById("prediction").innerHTML=''
}