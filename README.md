# Chat Completions API

This repository is an example of using OpenAI's chat completion to analyse a CSV file which may contain customer data such as financial information, the idea is that a user facing chatbot can then be built on top of this foundation, which will allow the customer to recieve financial information that is somewhat related pertaining to them.

Rather than using the OpenAI assistants API to use an assistant (e.g. a retrieval assistant) to analyse the CSV,
the CSV is injected into the prompt each time (though take caution when doing this regarding costs etc)

The repository demonstrates the use of injecting a name into the prompt to mimick a customer having their own chatbot account, though this likely would not be the a good path for a production application as jailbreaking may override the prompt which accomplishes such behavior.

Furthermore, the application makes little effort to account for infactual information, hallucinations etc.
