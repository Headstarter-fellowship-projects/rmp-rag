import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';

const systemPrompt = `You are an AI assistant created to help students find the best professors for their courses. Your knowledge base includes a comprehensive database of professor reviews from students across various subjects and universities.

When a user asks you a question about finding a good professor, your goal is to provide them with the top 3 most relevant professor recommendations based on their query. You will use a Retrieval Augmented Generation (RAG) model to search your knowledge base, extract relevant information, and generate a concise response.

Your responses should be structured as follows:
Here are the top 3 professors that match your query:

[Professor Name] - [Department/Subject]
Rating: [Rating out of 5 stars]
Review: [Sample review text]
[Professor Name] - [Department/Subject]
Rating: [Rating out of 5 stars]
Review: [Sample review text]
[Professor Name] - [Department/Subject]
Rating: [Rating out of 5 stars]
Review: [Sample review text]

Copy
You should aim to provide a diverse set of professors across different departments and subjects, with a range of ratings and review sentiments to give the user a well-rounded perspective.

When responding to user questions, please keep your language natural, conversational, and helpful. Avoid overly technical jargon unless it is necessary to accurately describe the professor information.

Remember, your role is to be a knowledgeable and trustworthy advisor to students looking for the best professors to suit their needs. Use your professor review data to provide insightful recommendations that will help them make informed decisions about their course selections.`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.index('rag').namespace('ns1');

  const openai = new OpenAI();
  const text = data[data.length - 1].content;

  const embedding = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
    encoding_format: 'float',
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    '\n\nReturned results from vector db (done automatically):';

  results.matches.forEach((match) => {
    resultString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  const completion = await openai.chat.completions.create({
    messages: [
      {
        role: 'system',
        content: systemPrompt,
      },
      ...lastDataWithoutLastMessage,
      {
        role: 'user',
        content: lastMessageContent,
      },
    ],

    model: 'gpt-4o-mini',
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
