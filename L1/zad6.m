prizes = randi(3, 10000, 1);
choices = randi(3, 10000, 1);

P_without_change = sum(prizes == choices) / 10000;

prizes_2 = randi(3, 10000, 1);
choices_2 = randi(3, 10000, 1);

P_with_change = sum(prizes_2 ~= choices_2) / 10000;
