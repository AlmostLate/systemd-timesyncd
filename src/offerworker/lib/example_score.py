from score import calculate_score_for_user, UsedOffer, User
offers = calculate_score_for_user(
    User(1, 1, 1, [UsedOffer(1, 123)]),
    [
        User(2, 1, 1, [UsedOffer(1, 122), UsedOffer(2, 122)]),
        User(3, 1, 1, [UsedOffer(1, 122), UsedOffer(2, 124)]),
    ]
)

print(offers)