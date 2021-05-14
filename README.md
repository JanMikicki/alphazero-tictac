# alphazero-tictac

These two notebooks implement Alpha Zero algorithm for a 3x3 Tic Tac Toe game.
Tabular case uses a table to store the state values and policy -- it is possible for a 3x3 board.
Neural cases uses a naural network to estmate those like the original Alpha Zero did obviously.
I would be possible to extend both to other board games relatively easy.

A good article explaining how Alpha Zero works is here: https://jonathan-hui.medium.com/alphago-zero-a-game-changer-14ef6e45eba5
MCTS implementation was based on this code example here: https://web.archive.org/web/20160308053456/http://mcts.ai/code/python.html

It is kind of hard to show learning progress because I don't have access to some ELO rating system.
Both implementations have some issue with capitalizing on a win at the very end -- I think the solution would be do decrease a number of MCTS iterations as the game goes on.
