# general plan 

- Pick some data set, i think imagenette works well here, should be nice and
  small. 
- import the data, make sure it works well with resnet to begin with. 
- implement a PGD attack, probably could do l2 and linf, see if linf is visible,
  and if not just go with linf. 
- design it so that there is a file that downloads data, a function that calls
  in image, function that attacks it, function that shows the adv_image and a
  function which loops over a bunch of them and returns some success rate. 
  