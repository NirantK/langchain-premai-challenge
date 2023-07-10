export default defineEventHandler(async (event) => {
	const config = useRuntimeConfig();
	let messages = [];
	const bodyData = await readBody(event);
	console.log('body_data', bodyData);
	// Check if body is empty. if not empty, then read the body's last array element. 
	// if the last element(an object) has a message property, then,
	// read the message property and assign it to the qdrant_prompt variable.
	let qdrantPrompt = '';
	if (bodyData.length > 0 && bodyData.slice(-1)[0].message) {
		qdrantPrompt = bodyData.slice(-1)[0].message;
	}
	console.log('qdrant_prompt', qdrantPrompt);

	// const previosMessages = await readBody(event);
	// messages = messages.concat(previosMessages);
	// let prompt =
	// 	messages.map((message) => `${message.role}: ${message.message}`).join('\n') + `\nAI:`;
	// console.log('prompt', prompt);
	// console.log("messages from ssr", messages);
	// const req = await fetch('https://api.openai.com/v1/completions', {
	// 	method: 'POST',
	// 	headers: {
	// 		'Content-Type': 'application/json',
	// 		Authorization: `Bearer ${config.OPENAI_API_KEY}`
	// 	},
	// 	body: JSON.stringify({
	// 		model: 'text-davinci-003',
	// 		prompt: prompt,
	// 		temperature: 0.9,
	// 		max_tokens: 512,
	// 		top_p: 1.0,
	// 		frequency_penalty: 0,
	// 		presence_penalty: 0.6,
	// 		stop: [' User:', ' AI:']
	// 	})
	// });

	// const res = await req.json();
	// const result = res.choices[0];
	// return {
	// 	message: result.text
	// };

	try {
		const req = await fetch('http://3.91.215.30:3000/ask', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				query: qdrantPrompt
			})
		});

		const response = await req.json();
		const response_body = JSON.parse(response.result.body);
		console.log('response from qdrant', response_body);
		return {
			message: response_body
		};
	} catch (error) {
		// Handle and log any errors
		console.error('Error:', error);
		// Return an appropriate error response
		return {
			message: `An error occurred while trying to fetch data from the API: ${error}`
		};
	}
});
