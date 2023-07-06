export default defineEventHandler(async (event) => {
	const config = useRuntimeConfig();
	let messages = [];
	const previosMessages = await readBody(event);
	console.log('previosMessages', previosMessages);
	messages = messages.concat(previosMessages);
	let prompt =
		messages.map((message) => `${message.role}: ${message.message}`).join('\n') + `\nAI:`;
	console.log('prompt', prompt);
	console.log("messages from ssr", messages)
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

	const req = await fetch('http://localhost:8000/ask', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
			},
		body: JSON.stringify({
			query: "any recipe on chocolate cake"
		})
		});
	const res = await req.json();
	// console.log('response from qdrant', res);
	return {
		message: res
	}
});
