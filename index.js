/* 
Rather than using the OpenAI API to upload a file and have 
the assistant (a GPT agent) parse the file, 
We could mimic this functionality as done below 
However the risk here is that the either 
- the file will need to be provided at the start of initiating a GPT instance 
    - when you go to ChatGPT, you can't guarantee the agent you are interacting with 
    is the agent you interacted with a week ago as your session may be different etc 
- or, even worse, the file (or, a representation of the file) will need to be provided during each prompt 

At first glance, using the completion API (rather than an assistant) seems easier at least in the short term, 
but if this were a project where longevity mattered, I'd figure out how to do this equivalent using the assistant 
API and the file upload endpoint. 

I briefly considered using the assistant API, but the API doc example uses the code interpreter tool, 
whereas I think we'd want to use the retrieval tool (honestly not sure if or which tool we'd use) 
And I think retrieval can become pretty costly, and honestly maybe overkill for our use case
- https://www.npmjs.com/package/openai#file-uploads
- https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
^ I'm assuming its too much work but it'd be interesting if there was an analysis of the differences in costs 
regarding using a (retrieval?) assistant and using prompt injection via the chat completion API 
e.g. injecting file contents and/or remembering context 
I'm not sure if the file uploads endpoint is priced differently etc, and offcourse it depends on the volume of text 
There's probably ways of "compressing" chunks of text into representable embeddings which require less tokens too 
There's then the question of how much does the quality of responses etc vary? 

*/
const fs = require('fs')
const csv = require('csv-parser')
// todo [refactor] decide whether to use ES6 module syntax or CommonJS 
const OpenAI = require('openai');

// optional implement Faker to generate data for the CSV file(s) - https://fakerjs.dev 
// ^ I've already done this using a Quadratic spreadsheet 

const openai = new OpenAI({
    // generally API keys should be kept secret e.g. with a .env file which is never commited to the repository etc 
    apiKey: '' // process.env['OPENAI_API_KEY'], // This is the default and can be omitted
});

/* 
is the OpenAI API strictly server side or can it be used client side 
though we may need to use BrowserifyJS 
alternatively we could just use JS fetch & transcribe the curl examples into fetch requests (which Phind can do) 
which gives us the added benefit of being able to use MockGPT to reduce development + testing based API calls 
by making sure the code works before we actually make the legit API call as opposed to doing this at the same time
*/ 

// Function to read and process CSV file
function processCsvFile(filePath) {
 return new Promise((resolve, reject) => {
    const results = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => results.push(data))
      .on('end', () => resolve(results))
      .on('error', (error) => reject(error));
 });
}

// Example question based on the CSV data
//const question = "What are William Porter's goals?"; // William Porter's financial goal is to "Save for retirement" and his non-financial goal is to "Travel to a new country".
//const question = "is a debt management plan right for Amy Holmes?" 
/* 
const question = "is a debt management plan right for Amy Holmes?" 
Based on the data provided, Amy Holmes has a positive bank balance and her expenses seem manageable compared to her income. She is currently not in debt and appears to have financial goals like investing in stocks and starting a hobby.
Therefore, based on the information available, a debt management plan does not seem necessary or suitable for Amy Holmes at this time.
Comment: I actually like this response 
*/
/* 
//const question = "who is in debt?" // may need to tweak the Faker script so it generates people who are debt 
To determine who is in debt, we need to calculate the total expenses for each individual and compare it to their bank balance.

1. William Porter's total expenses:
Total Expenses = Rent/Mortgage + Utilities + Groceries + Transport + Leisure
Total Expenses = 429.84 + 264.22 + 838.0 + 751.89 + 370.34
Total Expenses = 2654.29

2. Amy Holmes's total expenses:
Total Expenses = Rent/Mortgage + Utilities + Groceries + Transport + Leisure
Total Expenses = 116.31 + 191.32 + 589.87 + 150.29 + 115.11
Total Expenses = 1163.90

3. Jason Joseph's total expenses:
Total Expenses = Rent/Mortgage + Utilities + Groceries + Transport + Leisure
Total Expenses = 752.65 + 261.54 + 753.58 + 984.91 + 233.72
Total Expenses = 2986.40

Now, let's compare the total expenses to their bank balances:

1. William Porter: Bank Balance - Total Expenses
Bank Balance = 5852.34, Total Expenses = 2654.29
Bank Balance - Total Expenses = 3198.05 (Positive balance, not in debt)

2. Amy Holmes: Bank Balance - Total Expenses
Bank Balance = 7706.55, Total Expenses = 1163.90
Bank Balance - Total Expenses = 6542.65 (Positive balance, not in debt)

3. Jason Joseph: Bank Balance - Total Expenses
Bank Balance = 3857.56, Total Expenses = 2986.40
Bank Balance - Total Expenses = 871.16 (Positive balance, not in debt)

Based on the calculations, none of the individuals - William Porter, Amy Holmes, or Jason Joseph are in debt.
Comment: I suspect its going to be tedious confirming GPT's math :| 
There may also be ways of tricking GPT into not returning the same entire response as this could become costly 

When changing William Porter's bank balance from 5852.34 to 0, the response will correctly be:  
Therefore, William Porter is the one who is currently in debt.
*/ 

// testing name injection, the GPT should not respond with any data regarding William Porter
const question = "how much does William Porter spend on grocies?"
// GPT says:  Amy Holmes spends $589.87 on groceries.

// Process the CSV file 
// todo [refactor]: replace promises with async await syntax 
processCsvFile('customerdata.csv')
 .then(async data => {
    // Assuming you want to display the CSV data as a string for the prompt
    // You might want to format this differently based on your actual data structure
    const csvDataString = data.map(row => JSON.stringify(row)).join('\n');
    
    // Process the question and data to generate a prompt for the AI assistant
    // OpenAI API suggests using the unofficial 3rd party Tiktoken package for estimating token usage & costs 
    const prompt = `Given the following data:\n${csvDataString}\nQuestion: ${question}`;
    
    /* may be worth discussing rate limits in the actual report, though I don't think (& hope we don't) need 
    to worry about rate limits for our use case, assuming we use the API responsibly & sensibly (so please be cautious 
      doing API calls inside loops)
      https://cookbook.openai.com/examples/how_to_handle_rate_limits
      
      https://platform.openai.com/docs/guides/production-best-practices
      - also overkill for our use case but maybe something to mention in the report 
      
      There's also the concept of cacheing, though this has varying levels difficulty, 
      OpenAI may offer cache-ing via their API, local storage or IndexedDB could also be used for cache-ing 
      
      If we have an element of reproducibility (e.g. using the same seed) to get the same (or similar) responses, 
      then we could perhaps memoize (dynamic programming) questions & answers to avoid actually needing to make an
      API endpoint call & thus save costs 
      */
     const name = "Amy Holmes"
     const completion = await openai.chat.completions.create({
        // perhaps we'd want to consider sending the contextual/reference CSV file as the system message 
        // but I recall briefly reading that some models pay different degrees of attention to the system message  
        // maybe "You are a helpful finance assistant. Respond with empathy" or "You are a helpful finance and empathetic assistant"
        // it maybe niave, overkill & expensive, but one possible of way of mimicking memory is by sending all previous messages along with the new message 
        // 1 difference from the completion API and the assistant API is that assistants can remember past conversations 
        // So i wonder if the assistant agent is less ephemeral than the completion API agent 
        messages: [{"role": "system", "content": "You are a helpful finance assistant."},
        // should "name" be given inside of the question or inside the prompt? Where we could provide name during an initial request
            {"role": "user", "content": `${name} says '${prompt}', only provide data regarding ${name}`}],
        model: "gpt-3.5-turbo", // different models = different prices, probably just use GPT 3.5 as it'll be cheaper 
        // https://platform.openai.com/docs/api-reference/chat/create#chat-create-seed
        seed: 30
      });
      console.log(completion)
      // display this response somehow (e.g. as HTTP/Websocket/SSEvent response)
      console.log("GPT says: ", completion.choices[0].message.content)

   // ensure you're using the version 4 of the Openai NPM package so the JSON structure is consistent - https://stackoverflow.com/a/77246507
  //console.log(completion.choices[0]); // .completion.choices[0].content is the GPT's response 
})
 .catch(error => {
    console.error(error);
 });
