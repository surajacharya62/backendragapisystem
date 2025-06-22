Note: If this file is not showing the contents in the order than use code view mode.

I applied the various chunking methods and embedding models of different providers such openai, google and huggingface. Given below are some findings:

chunking-method || embedding-model || embedding-provider || embedding-latency || retrieval-latency
semantic           text-embedding-3-small  openai           1.82973 sec            1.27563 sec
semantic           embedding-001           google           3.293838               1.234849
semantic           all-MiniLM-L6-V2        huggingface      4.484629               0.509320
recursive          text-embedding-3-large  openai           2.012589               1.43847
recursive          embedding-001           google           2.895642               1.22742
recursive          all-MiniLM-L6-V2        huggingface      3.561253               0.785143  

text-embedding-3-small has 1536 vector dimensions, embedding-001 has 768 vector dimensions, all-MiniLM-L6-V2 has 384 vector dimension. While comparing with all methods text-embedding-3-small as well with large has better performance. However, remainning all other methods also shown equal performance but slight difference in overall embedding and retreival latency. 

For similarity search algorithms, consine has better accuracy and capture relevant semantic relations in nearest vectors. When I applied on various documents there were some difference in the metrics score while comparing with all consine,dotproduct and eulidean distance. Euclidean processes little slow when it tries to compare will vectors. Dotproduct shows some faster execution while comparing with consine. In overall, consine provides a better documents similarity. We can use each searchin technique best on the our requirements and application needs.
