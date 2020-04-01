var value_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0};
for (var i=0; i < data['Prediction'].length; i++){
	value_dict[data['Prediction'][i]] += 1;
}

var total = Object.values(value_dict).reduce((a,b) => a+b, 0)

var graph_data = [
  {
    x: ['Safe',
    'Texting with their Right Hand',
    'Talking on the phone with their Right Hand',
    'Texting with their Left Hand',
    'Talking on the phone with their Left Hand',
    'Operating the Radio',
    'Drinking',
    'Reaching Behind',
    'Focused on their Hair and Makeup',
    'Talking to a Passenger'],
    y: [value_dict[0]/total, value_dict[1]/total, value_dict[2]/total, value_dict[3]/total, value_dict[4]/total, value_dict[5]/total, value_dict[6]/total, value_dict[7]/total, value_dict[8]/total, value_dict[9]/total],
    type: 'bar',
    marker: {
    	color: 'rgb(141,2,31)'
    }
  }
];

var layout = {
	title: 'Distraction Broken Down by Percentage of Time',
	margin: {
		b: 200
	}
}

Plotly.newPlot('graph', graph_data, layout);